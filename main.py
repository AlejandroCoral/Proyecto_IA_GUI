import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================
# Matemática (SIN sklearn)
# =============================

def linear_regression_least_squares(x: np.ndarray, y: np.ndarray):
    """
    Regresión lineal simple por Mínimos Cuadrados.
    y = m*x + b
    """
    x = x.astype(float)
    y = y.astype(float)
    n = len(x)

    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.sum(x * y))
    sum_x2 = float(np.sum(x * x))

    denom = (n * sum_x2 - sum_x ** 2)
    if abs(denom) < 1e-12:
        raise ValueError("No se puede calcular la pendiente: X casi no varía (denominador ~ 0).")

    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n
    return m, b

def predict_linear(x_val: float, m: float, b: float) -> float:
    return m * x_val + b

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    return float(np.mean((y_true - y_pred) ** 2))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    return float(np.sqrt(np.sum((a - b) ** 2)))

def knn_predict(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, k: int):
    """
    KNN clasificación por votación mayoritaria.
    Distancia euclidiana.
    """
    if k <= 0:
        raise ValueError("K debe ser mayor que 0.")
    if k > len(X):
        raise ValueError("K no puede ser mayor que el número de datos.")

    distances = []
    for i in range(len(X)):
        d = euclidean_distance(X[i], x_new)
        distances.append((d, y[i]))

    distances.sort(key=lambda t: t[0])
    neighbors = distances[:k]

    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    # Desempate: si empatan, gana la clase del vecino más cercano
    max_vote = max(votes.values())
    tied = [lab for lab, v in votes.items() if v == max_vote]
    if len(tied) == 1:
        return tied[0]

    for d, lab in neighbors:
        if lab in tied:
            return lab


# =============================
# Utilidades GUI
# =============================

def try_read_csv(path: str):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El CSV está vacío.")
    return df

def parse_manual_csv(text: str):
    """
    Permite pegar datos tipo CSV en un TextBox.
    """
    text = text.strip()
    if not text:
        raise ValueError("No hay texto para procesar.")
    from io import StringIO
    return pd.read_csv(StringIO(text))

def fill_treeview(tree: ttk.Treeview, df: pd.DataFrame):
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for col in df.columns:
        tree.heading(col, text=str(col))
        tree.column(col, width=120, anchor="center")
    for _, row in df.iterrows():
        tree.insert("", "end", values=[row[c] for c in df.columns])

def safe_float(s: str, name: str):
    try:
        return float(s)
    except:
        raise ValueError(f"'{name}' debe ser numérico.")


# =============================
# App principal
# =============================

class IAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        style = ttk.Style(self)
        style.theme_use("vista")
        style.configure("TNotebook.Tab", padding=[12,6])
        self.configure(bg="#eef9f2")
        style.configure("Hint.TLabel", font=("Segoe UI", 10))
      
        self.title("Mini laboratorio de IA – Regresión Lineal y KNN")
        self.geometry("1200x720")

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 12, "bold"),
            foreground="#2c3e50"
        )

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8
        )

        welcome = ttk.Label(
            self,
            text="Mini laboratorio para experimentar con algoritmos básicos de Inteligencia Artificial",
            font=("Segoe UI", 10)
        )
        welcome.pack(pady=5)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.reg_frame = ttk.Frame(self.notebook, padding=10)
        self.knn_frame = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.reg_frame, text="Regresión Lineal")
        self.notebook.add(self.knn_frame, text="KNN Clasificación")

        self.reg_df = None
        self.knn_df = None

        self._build_regression_ui()
        self._build_knn_ui()

    # -------------------------
    # UI Regresión
    # -------------------------
    def _build_regression_ui(self):
        left = ttk.Frame(self.reg_frame)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Frame(self.reg_frame)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(left, text="Módulo de carga (Regresión)", style="Title.TLabel").pack(anchor="w")
        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Cargar CSV", command=self.load_regression_csv).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Usar datos pegados", command=self.load_regression_manual).pack(side="left", padx=5)

        ttk.Label(left, text="Pega datos tipo CSV (Regresión):").pack(anchor="w", pady=(10, 2))
        self.reg_manual_text = tk.Text(left, width=45, height=10)
        self.reg_manual_text.pack(fill="x")
        self.reg_manual_text.insert("1.0", "X,Y\n1,52\n2,55\n3,61\n4,66\n5,72\n")

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="Predicción", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        pred_row = ttk.Frame(left)
        pred_row.pack(fill="x", pady=6)
        ttk.Label(pred_row, text="X:").pack(side="left")
        self.reg_pred_x = ttk.Entry(pred_row, width=12)
        self.reg_pred_x.pack(side="left", padx=6)
        ttk.Button(pred_row, text="Predecir Y", command=self.regression_predict).pack(side="left")

        self.reg_result_label = ttk.Label(left, text="Ecuación: —\nMSE: —\nPredicción: —", justify="left")
        self.reg_result_label.pack(anchor="w", pady=10)

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Button(left, text="Entrenar / Calcular modelo", command=self.train_regression, style="Accent.TButton").pack(fill="x", pady=6)
        ttk.Label(right, text="Datos cargados", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.reg_tree = ttk.Treeview(right, height=10)
        self.reg_tree.pack(fill="x", pady=6)

        ttk.Label(right, text="Gráfico (scatter + línea)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 0))
        self.reg_fig = Figure(figsize=(6.5, 4.0), dpi=100)
        self.reg_ax = self.reg_fig.add_subplot(111)
        self.reg_canvas = FigureCanvasTkAgg(self.reg_fig, master=right)
        self.reg_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.reg_m = None
        self.reg_b = None

    def load_regression_csv(self):
        path = filedialog.askopenfilename(
            title="Selecciona CSV para Regresión",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            df = try_read_csv(path)
            self._validate_regression_df(df)
            self.reg_df = df
            fill_treeview(self.reg_tree, df)
            self.reg_result_label.config(text="Ecuación: —\nMSE: —\nPredicción: —")
            self._plot_regression(None)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el CSV:\n{e}")

    def load_regression_manual(self):
        try:
            df = parse_manual_csv(self.reg_manual_text.get("1.0", "end"))
            self._validate_regression_df(df)
            self.reg_df = df
            fill_treeview(self.reg_tree, df)
            self.reg_result_label.config(text="Ecuación: —\nMSE: —\nPredicción: —")
            self._plot_regression(None)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar el texto:\n{e}")

    def _validate_regression_df(self, df: pd.DataFrame):
        if df.shape[1] != 2:
            raise ValueError("Para regresión, el CSV debe tener EXACTAMENTE 2 columnas.")
        df.columns = ["X", "Y"]
        df["X"] = pd.to_numeric(df["X"], errors="raise")
        df["Y"] = pd.to_numeric(df["Y"], errors="raise")

    def train_regression(self):
        if self.reg_df is None:
            messagebox.showwarning("Falta datos", "Carga o pega datos para entrenar la regresión.")
            return
        try:
            x = self.reg_df["X"].to_numpy()
            y = self.reg_df["Y"].to_numpy()
            m, b = linear_regression_least_squares(x, y)
            yhat = m * x + b
            err = mse(y, yhat)

            self.reg_m, self.reg_b = m, b
            self.reg_result_label.config(
                text=f"Ecuación: Y = {m:.6f}X + {b:.6f}\nMSE: {err:.6f}\nPredicción: —"
            )
            self._plot_regression((m, b))
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo entrenar el modelo:\n{e}")

    def regression_predict(self):
        if self.reg_m is None or self.reg_b is None:
            messagebox.showwarning("Modelo no entrenado", "Primero entrena/calcula el modelo.")
            return
        try:
            x_val = safe_float(self.reg_pred_x.get().strip(), "X")
            y_val = predict_linear(x_val, self.reg_m, self.reg_b)

            prev = self.reg_result_label.cget("text").splitlines()
            eq = prev[0] if len(prev) > 0 else "Ecuación: —"
            mse_line = prev[1] if len(prev) > 1 else "MSE: —"
            self.reg_result_label.config(text=f"{eq}\n{mse_line}\nPredicción: Y({x_val}) = {y_val:.6f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_regression(self, model):
        self.reg_ax.clear()
        self.reg_ax.set_title("Regresión Lineal Simple")
        self.reg_ax.set_xlabel("X")
        self.reg_ax.set_ylabel("Y")

        if self.reg_df is not None:
            x = self.reg_df["X"].to_numpy()
            y = self.reg_df["Y"].to_numpy()
            self.reg_ax.scatter(x, y)

            if model is not None:
                m, b = model
                xs = np.linspace(np.min(x), np.max(x), 100)
                ys = m * xs + b
                self.reg_ax.plot(xs, ys)

        self.reg_canvas.draw()

    # -------------------------
    # UI KNN
    # -------------------------
    def _build_knn_ui(self):
        left = ttk.Frame(self.knn_frame)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Frame(self.knn_frame)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(left, text="Módulo de carga (KNN)", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Cargar CSV", command=self.load_knn_csv).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Usar datos pegados", command=self.load_knn_manual).pack(side="left", padx=5)

        ttk.Label(left, text="Pega datos tipo CSV (KNN):").pack(anchor="w", pady=(10, 2))
        self.knn_manual_text = tk.Text(left, width=45, height=10)
        self.knn_manual_text.pack(fill="x")
        self.knn_manual_text.insert("1.0", "x1,x2,label\n1,1,A\n1,2,A\n6,6,B\n7,7,B\n3,6,C\n4,7,C\n")

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="Configuración / Predicción", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        k_row = ttk.Frame(left)
        k_row.pack(fill="x", pady=6)
        ttk.Label(k_row, text="K:").pack(side="left")
        self.knn_k = ttk.Spinbox(k_row, from_=1, to=99, width=8)
        self.knn_k.pack(side="left", padx=6)
        self.knn_k.set("3")

        p_row = ttk.Frame(left)
        p_row.pack(fill="x", pady=6)
        ttk.Label(p_row, text="x1:").pack(side="left")
        self.knn_x1 = ttk.Entry(p_row, width=10)
        self.knn_x1.pack(side="left", padx=4)
        ttk.Label(p_row, text="x2:").pack(side="left")
        self.knn_x2 = ttk.Entry(p_row, width=10)
        self.knn_x2.pack(side="left", padx=4)
        ttk.Button(p_row, text="Clasificar", command=self.knn_classify).pack(side="left", padx=6)

        self.knn_result_label = ttk.Label(left, text="Clase predicha: —", justify="left")
        self.knn_result_label.pack(anchor="w", pady=10)

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Button(left, text="Graficar datos KNN", command=self.plot_knn).pack(fill="x", pady=4)

        ttk.Label(right, text="Datos cargados", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.knn_tree = ttk.Treeview(right, height=10)
        self.knn_tree.pack(fill="x", pady=6)

        ttk.Label(right, text="Gráfico (puntos por clase)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 0))
        self.knn_fig = Figure(figsize=(6.5, 4.0), dpi=100)
        self.knn_ax = self.knn_fig.add_subplot(111)
        self.knn_canvas = FigureCanvasTkAgg(self.knn_fig, master=right)
        self.knn_canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_knn_csv(self):
        path = filedialog.askopenfilename(
            title="Selecciona CSV para KNN",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            df = try_read_csv(path)
            self._validate_knn_df(df)
            self.knn_df = df
            fill_treeview(self.knn_tree, df)
            self.knn_result_label.config(text="Clase predicha: —")
            self.plot_knn()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el CSV:\n{e}")

    def load_knn_manual(self):
        try:
            df = parse_manual_csv(self.knn_manual_text.get("1.0", "end"))
            self._validate_knn_df(df)
            self.knn_df = df
            fill_treeview(self.knn_tree, df)
            self.knn_result_label.config(text="Clase predicha: —")
            self.plot_knn()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar el texto:\n{e}")

    def _validate_knn_df(self, df: pd.DataFrame):
        if df.shape[1] != 3:
            raise ValueError("Para KNN, el CSV debe tener EXACTAMENTE 3 columnas: x1,x2,label.")
        df.columns = ["x1", "x2", "label"]
        df["x1"] = pd.to_numeric(df["x1"], errors="raise")
        df["x2"] = pd.to_numeric(df["x2"], errors="raise")
        df["label"] = df["label"].astype(str)

    def knn_classify(self):
        if self.knn_df is None:
            messagebox.showwarning("Falta datos", "Carga o pega datos para usar KNN.")
            return
        try:
            k = int(self.knn_k.get())
            x1 = safe_float(self.knn_x1.get().strip(), "x1")
            x2 = safe_float(self.knn_x2.get().strip(), "x2")

            X = self.knn_df[["x1", "x2"]].to_numpy()
            y = self.knn_df["label"].to_numpy()
            pred = knn_predict(X, y, np.array([x1, x2], dtype=float), k)

            self.knn_result_label.config(text=f"Clase predicha: {pred}")
            self._plot_knn_point(x1, x2)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_knn(self):
        self.knn_ax.clear()
        self.knn_ax.set_title("KNN — Puntos por clase")
        self.knn_ax.set_xlabel("x1")
        self.knn_ax.set_ylabel("x2")

        if self.knn_df is not None:
            for label, group in self.knn_df.groupby("label"):
                self.knn_ax.scatter(group["x1"], group["x2"], label=str(label))
            self.knn_ax.legend()

        self.knn_canvas.draw()

    def _plot_knn_point(self, x1, x2):
        self.plot_knn()
        self.knn_ax.scatter([x1], [x2], marker="x", s=120)
        self.knn_canvas.draw()


if __name__ == "__main__":
    app = IAApp()
    app.mainloop()