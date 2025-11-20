#!/usr/bin/env python3
"""
3D Keypoints Labeling Application
Main entry point for the application
"""

import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Set dark theme style (VSCode-like)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
            color: #cccccc;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #cccccc;
            padding: 8px 16px;
            border: 1px solid #3c3c3c;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            border-bottom: 2px solid #007acc;
        }
        QTabBar::tab:hover {
            background-color: #2d2d2d;
        }
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #0a4d73;
        }
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666666;
        }
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            padding: 4px;
            border-radius: 2px;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #007acc;
        }
        QLabel {
            color: #cccccc;
        }
        QComboBox {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            padding: 4px;
            border-radius: 2px;
        }
        QComboBox:hover {
            border: 1px solid #007acc;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #252526;
            color: #cccccc;
            selection-background-color: #007acc;
            border: 1px solid #3c3c3c;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3c3c3c;
            height: 8px;
            background: #252526;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #007acc;
            border: 1px solid #007acc;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #1177bb;
        }
        QScrollBar:vertical {
            background-color: #1e1e1e;
            width: 12px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #424242;
            min-height: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #4e4e4e;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

