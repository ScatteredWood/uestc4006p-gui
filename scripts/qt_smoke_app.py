from PySide6.QtWidgets import QApplication, QWidget


def main() -> int:
    app = QApplication([])
    w = QWidget()
    w.setWindowTitle('qt smoke')
    w.resize(320, 120)
    w.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
