import sys
import pandas as pd
import webbrowser
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QHBoxLayout, QMessageBox
)

from graph_core import (
    count_pairs, filter_dictionary_by_value,
    create_interactive_graph, export_graph_png, CLASS_MAP
)

class GraphGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Граф связей БАД - Desktop GUI")
        self.resize(650, 350)

        self.df = None
        layout = QVBoxLayout()

        self.status = QLabel("Файл не загружен")
        layout.addWidget(self.status)

        btn_load = QPushButton("Загрузить Excel/CSV")
        btn_load.clicked.connect(self.load_file)
        layout.addWidget(btn_load)

        self.col_box = QComboBox()
        layout.addWidget(QLabel('Выберите столбец:'))
        layout.addWidget(self.col_box)

        h = QHBoxLayout()
        h.addWidget(QLabel("Порог веса ребер:"))
        self.weight = QSpinBox()
        self.weight.setRange(1, 500)
        h.addWidget(self.weight)
        layout.addLayout(h)

        btn_html = QPushButton("Построить интерактивный граф (HTML)")
        btn_html.clicked.connect(self.build_html)
        layout.addWidget(btn_html)

        btn_png = QPushButton("Построить PNG граф")
        btn_png.clicked.connect(self.build_png)
        layout.addWidget(btn_png)

        self.setLayout(layout)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Excel/CSV (*.xlsx *.csv)")
        if not path: return

        self.df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
        self.status.setText("Загружено: " + path)
        self.col_box.clear()
        self.col_box.addItems(self.df.columns)

    def build_pairs(self):
        if self.df is None:
            QMessageBox.warning(self,"Ошибка","Зарузите данные!")
            return None
        col = self.col_box.currentText()
        pairs = count_pairs(self.df, col, mapper=lambda x: CLASS_MAP.get(x, x))
        return filter_dictionary_by_value(pairs, self.weight.value())
        
    def build_html(self):
        pairs = self.build_pairs()
        if not pairs: return
        try:
            create_interactive_graph(pairs, "graph.html")
            QMessageBox.information(self,"Готово","Интерактив сохранен: graph.html")
            webbrowser.open("graph.html")
        except Exception as e:
            import traceback
            with open("error_log.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            QMessageBox.showerror("Ошибка", "Смотри error_log.txt")

    def build_png(self):
        pairs = self.build_pairs()
        if not pairs: return
        export_graph_png(pairs, "graph.png", min_weight=self.weight.value())
        QMessageBox.information(self, "Готово", "PNG сохранен: graph.png")
        webbrowser.open("graph.png")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GraphGUI()
    gui.show()
    sys.exit(app.exec_())