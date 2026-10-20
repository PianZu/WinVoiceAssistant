"""
overlay_widget.py

The floating "pill" overlay: a frameless, translucent, always-on-top widget
pinned to the top-center of the primary monitor. It never steals keyboard
focus from whatever app the user is working in.

Hidden by default - it's only revealed when PTT engages or a wake word is
heard (flash_active()), and auto-hides again shortly after listening stops
or an action's confirmation badge has had its moment on screen. Right-click
for a "Hide" / "Exit Assistant" menu.

States:
    idle       - small pill, dim ambient pulse (may show a hotkey hint) -
                 hides itself shortly after, unless an action just fired
    active     - wake word / push-to-talk just engaged: instant vivid cyan
                 pop with a "JARVIS LISTENING" badge, no ramp-up
    listening  - same cyan accent, waiting for the first words
    streaming  - pill widens to show the live interim transcript
    action     - an emerald confirmation badge (e.g. "[LAUNCHED ARC]") is
                 shown, then the pill auto-hides
"""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtCore import Qt, QTimer, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QFont, QFontMetrics
from PyQt6.QtWidgets import QApplication, QMenu, QWidget

PILL_HEIGHT = 44
PILL_MIN_WIDTH = 140
PILL_MAX_WIDTH = 640
PILL_PADDING_X = 20
TOP_MARGIN = 14

BG_COLOR = QColor(20, 20, 25, int(255 * 0.88))
BORDER_COLOR = QColor(255, 255, 255, 22)
TEXT_COLOR = QColor(235, 235, 240, 255)
IDLE_COLOR = QColor(120, 170, 255, 255)  # dim ambient blue while asleep
CYAN_ACCENT = QColor(0, 229, 255, 255)  # vivid neon cyan while actively listening
EMERALD_COLOR = QColor(0, 224, 122, 255)  # confirmation-badge green

ACTION_BADGE_LIFETIME_MS = 1500
IDLE_HIDE_DELAY_MS = 300
ACTIVE_POP_HINT = "JARVIS LISTENING"


class PillState(Enum):
    IDLE = auto()
    ACTIVE = auto()  # wake word / PTT just engaged - instant pop, no ramp-up
    LISTENING = auto()
    STREAMING = auto()
    ACTION = auto()


class PillOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._state = PillState.IDLE
        self._transcript = ""
        self._action_label = ""
        self._active_hint = ACTIVE_POP_HINT
        self._idle_hint = ""  # e.g. "Hold Right Alt to talk" - set by main.py
        self._glow = 0.4  # 0..1, driven by the pulse animation

        self._font = QFont("Segoe UI", 11)
        self._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

        self._pulse_anim = QPropertyAnimation(self, b"glow")
        self._pulse_anim.setDuration(1400)
        self._pulse_anim.setStartValue(0.25)
        self._pulse_anim.setEndValue(0.9)
        self._pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.finished.connect(self._reverse_pulse)
        self._pulse_direction_up = True

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        self._resize_to_content()
        self._reposition()
        self._pulse_anim.start()
        self.hide()  # only revealed on PTT engage / wake word - see flash_active()

    # -- glow property (animated) -----------------------------------------

    def _get_glow(self) -> float:
        return self._glow

    def _set_glow(self, value: float) -> None:
        self._glow = value
        self.update()

    glow = pyqtProperty(float, _get_glow, _set_glow)

    def _reverse_pulse(self) -> None:
        # QPropertyAnimation with loopCount(-1) already loops start->end; we
        # ping-pong manually so the glow breathes in and out.
        #
        # The `finished` signal can still be delivered from Qt's queue during
        # app teardown, after the animation's underlying C++ object has
        # already been deleted (closeEvent's stop() only prevents *new*
        # emissions, not ones already queued) — accessing it then raises
        # RuntimeError: wrapped C/C++ object ... has been deleted.
        try:
            start, end = self._pulse_anim.startValue(), self._pulse_anim.endValue()
        except RuntimeError:
            return
        self._pulse_anim.setStartValue(end)
        self._pulse_anim.setEndValue(start)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._pulse_anim.stop()
        super().closeEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        hide_action = menu.addAction("Hide")
        exit_action = menu.addAction("Exit Assistant")
        chosen = menu.exec(event.globalPos())
        if chosen == hide_action:
            self._hide_timer.stop()
            self.hide()
        elif chosen == exit_action:
            QApplication.instance().quit()

    # -- public API (called from the intent/voice pipeline) ----------------

    def set_idle_hint(self, hint: str) -> None:
        """A short hotkey hint shown while idle, e.g. "Hold Right Alt to talk"."""
        self._idle_hint = hint
        if self._state == PillState.IDLE:
            self._resize_to_content()
            self.update()

    @pyqtSlot(str)
    def flash_active(self, hint: str = ACTIVE_POP_HINT) -> None:
        """Wake word detected / PTT engaged: pop straight to full cyan glow,
        no ramp-up, so the reaction reads as instant rather than animated.

        Reveals the pill (hidden by default) and cancels any pending
        auto-hide, so re-engaging never races a stale hide() from a
        previous cycle into hiding the pill right after it's shown again.
        """
        self._hide_timer.stop()
        self._active_hint = hint
        self._state = PillState.ACTIVE
        self._glow = 1.0
        self._resize_to_content()
        self.show()
        self.update()

    @pyqtSlot(bool)
    def set_listening(self, listening: bool) -> None:
        if listening:
            self._hide_timer.stop()
            self._state = PillState.LISTENING
            self.show()
        else:
            self._transcript = ""
            if self._state == PillState.ACTION:
                # An action badge is currently showing (and already has its
                # own hide timer running via show_action) - don't cut its
                # display short just because listening stopped.
                pass
            else:
                self._state = PillState.IDLE
                self._hide_timer.start(IDLE_HIDE_DELAY_MS)
        self._resize_to_content()
        self.update()

    @pyqtSlot(str)
    def update_transcript(self, text: str) -> None:
        self._transcript = text
        self._state = PillState.STREAMING if text else PillState.LISTENING
        self._resize_to_content()
        self.update()

    @pyqtSlot(str)
    def show_action(self, label: str) -> None:
        self._hide_timer.stop()
        self._action_label = label
        self._state = PillState.ACTION
        self._resize_to_content()
        self.show()
        self.update()
        self._hide_timer.start(ACTION_BADGE_LIFETIME_MS)

    # -- layout --------------------------------------------------------

    def _content_text(self) -> str:
        if self._state == PillState.ACTION:
            return self._action_label
        if self._state == PillState.ACTIVE:
            return self._active_hint
        if self._state in (PillState.STREAMING, PillState.LISTENING) and self._transcript:
            return self._transcript
        if self._state == PillState.LISTENING:
            return "Listening..."
        return self._idle_hint or "●"

    def _resize_to_content(self) -> None:
        metrics = QFontMetrics(self._font)
        text = self._content_text()
        text_width = metrics.horizontalAdvance(text)
        width = max(PILL_MIN_WIDTH, min(PILL_MAX_WIDTH, text_width + PILL_PADDING_X * 2 + 28))
        self.resize(width, PILL_HEIGHT)
        self._reposition()

    def _reposition(self) -> None:
        screen = QApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + TOP_MARGIN
        self.move(x, y)

    # -- painting --------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height())
        path = QPainterPath()
        path.addRoundedRect(rect, self.height() / 2, self.height() / 2)

        painter.fillPath(path, BG_COLOR)
        painter.setPen(BORDER_COLOR)
        painter.drawPath(path)

        self._paint_mic_glyph(painter)
        self._paint_text(painter)

    def _accent_color(self) -> QColor:
        if self._state == PillState.ACTION:
            return EMERALD_COLOR
        if self._state in (PillState.ACTIVE, PillState.LISTENING, PillState.STREAMING):
            return CYAN_ACCENT
        return IDLE_COLOR

    def _paint_mic_glyph(self, painter: QPainter) -> None:
        cx, cy = 24, self.height() / 2
        pulsing = self._state in (PillState.LISTENING, PillState.STREAMING)
        radius = 5 + (4 if pulsing else 0) * self._glow

        color = self._accent_color()
        glow_color = QColor(color)
        glow_color.setAlpha(int(60 * self._glow) if self._state in (PillState.IDLE, PillState.LISTENING, PillState.STREAMING) else 220)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow_color)
        painter.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

        painter.setBrush(color)
        painter.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))

    def _paint_text(self, painter: QPainter) -> None:
        painter.setFont(self._font)
        color = EMERALD_COLOR if self._state == PillState.ACTION else \
            (CYAN_ACCENT if self._state == PillState.ACTIVE else TEXT_COLOR)
        painter.setPen(color)
        text_rect = QRectF(44, 0, self.width() - 44 - PILL_PADDING_X, self.height())
        metrics = QFontMetrics(self._font)
        elided = metrics.elidedText(self._content_text(), Qt.TextElideMode.ElideLeft, int(text_rect.width()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)
