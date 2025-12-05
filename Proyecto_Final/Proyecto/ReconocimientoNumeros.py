import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
from PIL import Image, ImageTk, ImageGrab
from datetime import datetime

from ML_Predict import ML_Predict


class ReconocimientoNumeros:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Digitos")
        self.root.geometry("900x700")

        self.dataset = []
        self.MLPredict = ML_Predict(
            "saved_images", "imagenesnumeros_data.npz", "reconocimiento_digitos_svm.pkl")
        self.create_widgets()
        self.create_ml_controls()

    def create_widgets(self):
        # Título principal
        title_label = tk.Label(self.root, text="Reconocimiento de Digitos",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Frame para controles principales
        main_controls_frame = tk.Frame(self.root)
        main_controls_frame.pack(pady=10)

        # Frame principal para contenido
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Frame izquierdo para imágenes
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame para la imagen procesada
        processed_frame = tk.LabelFrame(left_frame, text="Número a procesar")
        processed_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        drawing_frame = tk.Frame(processed_frame)
        drawing_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(drawing_frame, bg="white",
                                cursor="pencil")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Frame derecho para controles y lista
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # Frame para controles de etiquetado
        labeling_frame = tk.LabelFrame(right_frame, text="Imagen seleccionada")
        labeling_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Selected image display
        image_display_frame = tk.Frame(labeling_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        self.original_image_label = tk.Label(image_display_frame, text="No hay imagen cargada",
                                             bg="white", relief="solid", bd=1)
        self.original_image_label.pack(
            padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Frame para la lista de imágenes
        images_frame = tk.LabelFrame(right_frame, text="Dataset")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview para mostrar el dataset
        self.dataset_tree = ttk.Treeview(
            images_frame, columns=("Imagen", "Etiqueta"), show="headings")
        self.dataset_tree.heading("Imagen", text="Imagen")
        self.dataset_tree.heading("Etiqueta", text="Etiqueta")
        self.dataset_tree.column("Imagen", width=150)
        self.dataset_tree.column("Etiqueta", width=80)

        scrollbar = ttk.Scrollbar(
            images_frame, orient="vertical", command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=scrollbar.set)

        self.dataset_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_tree.bind("<Double-1>", self.on_image_select)
        # Load existing images
        self.load_images()
        self.MLPredict.load_and_process_images()

    def on_image_select(self, event):
        selected_item = self.dataset_tree.selection()

        if selected_item:
            item = selected_item[0]
            valores = self.dataset_tree.item(item, 'values')
            image = valores[0]
            path = f"saved_images/{image[0]}/{valores[0]}"
            self.display_original_image(path)

    def create_ml_controls(self):
        # Frame para controles de ML
        ml_frame = tk.LabelFrame(self.root, text="Proceso")
        ml_frame.pack(fill=tk.X, padx=20, pady=10)

        ml_controls_frame = tk.Frame(ml_frame)
        ml_controls_frame.pack(pady=10)

        # Botón para entrenar modelo
        clean_button = tk.Button(ml_controls_frame, text="Limpiar pantalla",
                                 command=self.clean_image, bg="lightblue", width=15)
        clean_button.pack(side=tk.LEFT, padx=5)

        train_button = tk.Button(ml_controls_frame, text="Entrenar Modelo",
                                 command=self.train_model, bg="lightgreen", width=15)
        train_button.pack(side=tk.LEFT, padx=5)

        # Botón para predecir
        predict_button = tk.Button(ml_controls_frame, text="Predecir",
                                   command=self.predict_image, bg="lightyellow", width=15)
        predict_button.pack(side=tk.LEFT, padx=5)

        # Área de información del modelo
        self.model_info_label = tk.Label(ml_frame, text="Modelo: No entrenado",
                                         font=("Arial", 10), bg="white", relief="solid", bd=1)
        self.model_info_label.pack(fill=tk.X, padx=10, pady=5)

    def display_original_image(self, image_path):
        try:
            # Leer imagen con OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo cargar la imagen")

            # Convertir BGR a RGB para mostrar
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Redimensionar para mostrar
            display_image = self.resize_image_for_display(image_rgb, 100)

            # Convertir para Tkinter
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)

            # Actualizar el label
            self.original_image_label.configure(image=photo, text="")
            self.original_image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {e}")

    def load_images(self):
        """Load existing images from saved_images directory using os.walk"""
        base_path = 'saved_images'

        if os.path.exists(base_path):
            band = True

            for root, dirs, files in os.walk(base_path):
                # El label será el nombre de la última carpeta
                label = os.path.basename(root)

                # Solo procesar carpetas con nombres de 0-9
                if label.isdigit():
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(root, file)
                            self.dataset_tree.insert(
                                "", "end", values=(file, label))

                            if band:
                                self.display_original_image(file_path)
                                band = False

    def train_model(self):
        elementos = self.dataset_tree.get_children()
        if len(elementos) < 15:
            messagebox.showwarning(
                "Advertencia", "Se necesitan al menos 15 muestras para entrenar")
            return

        try:
            accurancy, y_Pred = self.MLPredict.Entrenar_Modelo()
            self.model_info_label.config(
                text=f"Modelo Entrenado"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error en el entrenamiento: {e}")

    def clean_image(self):
        self.canvas.delete("all")

    def predict_image(self):
        if self.MLPredict.Model is None:
            messagebox.showwarning("Advertencia", "Primero entrena un modelo")
            return

        try:
            image = self.save_drawing()
            predict = self.MLPredict.Predecir_Digito(image)
            messagebox.showinfo("Predicción",
                                f"Número reconocido: {predict}")
            os.remove(image)

        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {e}")

    def resize_image_for_display(self, image, max_size):
        """Redimensionar imagen para mostrar en la interfaz"""
        h, w = image.shape[:2]

        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))

        return cv2.resize(image, (new_w, new_h))

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y,
                                        event.x, event.y,
                                        width=8, capstyle=tk.ROUND,
                                        smooth=tk.TRUE, fill="black")
            self.last_x = event.x
            self.last_y = event.y

    def stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_image = None

    def save_drawing(self):
        # Save the image
        if not os.path.exists('predict_images'):
            os.makedirs('predict_images')
        fecha_hora = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = f"predict_images/predict_{fecha_hora}.png"

        # Convert canvas to image using PIL
        self.save_canvas_as_image(filename)
        return filename

    def save_canvas_as_image(self, filename):
        """Guardar canvas directamente usando ImageGrab"""
        try:
            # Obtener las coordenadas del canvas en pantalla
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()

            # Capturar la región de la pantalla
            img = ImageGrab.grab(bbox=(x, y, x1, y1))
            img.save(filename, 'PNG')

            print(f"Imagen guardada: {filename}")
            return True

        except Exception as e:
            print(f"Error con ImageGrab: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ReconocimientoNumeros(root)
    root.mainloop()
