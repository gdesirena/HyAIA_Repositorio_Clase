import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageGrab
import os


class CapturarNumeros:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Digitos a mano")
        self.root.geometry("900x750")
        self.root.configure(bg="#F5F5DC")  # Beige background

        # Variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.image_list = []
        self.selected_image = None
        self.canvas_image = None
        self.model_trained = False

        # Create main container
        self.create_widgets()

    def create_widgets(self):
        # Top bar (blue)
        top_bar = tk.Frame(self.root, bg="#4169E1", height=40)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        title_label = tk.Label(top_bar, text="Captura imagenes de numeros para utilizarlos para entrenamiento de IA",
                               bg="#4169E1", fg="white",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=8)

       # Main content area
        main_frame = tk.Frame(self.root, bg="#F5F5DC")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel
        left_panel = tk.Frame(main_frame, bg="#F5F5DC")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Drawing area
        drawing_frame = tk.Frame(left_panel, bg="#F5F5DC")
        drawing_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(drawing_frame, bg="white", width=400, height=400,
                                cursor="pencil")
        self.canvas.pack(pady=10)

        # Label for drawing area
        draw_label = tk.Label(drawing_frame, text="Dibuja un número",
                              bg="#F5F5DC", font=("Arial", 12))
        draw_label.pack()

        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Input field
        input_frame = tk.Frame(left_panel, bg="#F5F5DC")
        input_frame.pack(fill=tk.X, pady=10)

        input_label = tk.Label(input_frame, text="Número:",
                               bg="#F5F5DC", font=("Arial", 10))
        input_label.pack(side=tk.LEFT, padx=(0, 5))

        self.input_entry = tk.Entry(input_frame, font=("Arial", 10), width=30)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        button_frame = tk.Frame(left_panel, bg="#F5F5DC")
        button_frame.pack(fill=tk.X, pady=10)

        self.save_button = tk.Button(button_frame, text="Guardar",
                                     command=self.save_drawing,
                                     bg="#4CAF50", fg="white",
                                     font=("Arial", 11, "bold"),
                                     padx=20, pady=5)
        self.save_button.pack(fill=tk.X, pady=2)


        # Clear button (additional functionality)
        self.clear_button = tk.Button(button_frame, text="Limpiar",
                                      command=self.clear_canvas,
                                      bg="#FF9800", fg="white",
                                      font=("Arial", 10),
                                      padx=20, pady=5)
        self.clear_button.pack(fill=tk.X, pady=2)

        
        # Right panel
        right_panel = tk.Frame(main_frame, bg="#F5F5DC", width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)

        # Selected image display
        image_display_frame = tk.Frame(right_panel, bg="#F5F5DC")
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        image_label_text = tk.Label(image_display_frame, text="Imagen seleccionada",
                                    bg="#F5F5DC", font=("Arial", 10, "bold"))
        image_label_text.pack()

        self.image_display = tk.Canvas(image_display_frame, bg="white",
                                       width=280, height=280)
        self.image_display.pack(pady=5)

        # Image list
        list_frame = tk.Frame(right_panel, bg="#F5F5DC")
        list_frame.pack(fill=tk.BOTH, expand=True)

        list_label = tk.Label(list_frame, text="Imagenes",
                              bg="#F5F5DC", font=("Arial", 10, "bold"))
        list_label.pack()

        # Listbox with scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame,
                                        yscrollcommand=scrollbar.set,
                                        font=("Arial", 10),
                                        height=8)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        # Bind listbox selection
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Load existing images
        self.load_images()

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
        base_path = 'saved_images'
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        input_value = self.input_entry.get()
        count = 0

        for image_path in self.image_list:
            # Obtener el nombre del archivo sin la carpeta
            filename = image_path.split('\\')[-1]
            # Extraer el número antes del '_'
            number = filename.split('_')[0]
            if number ==input_value:
                count += 1

        
        next_number = count + 1
        next_str = f"{next_number:02d}"

        print(f"count{count} next ={next_number}, next_str{next_str}")
        filename = f"{base_path}/{input_value}/{input_value}_{next_str}.jpeg"

        # Convert canvas to image using PIL
        self.save_canvas_as_image(filename)

        # Add to list
        self.image_list.append(filename)
        self.image_listbox.insert(tk.END, f"Imagen {len(self.image_list)}")
        self.canvas.delete("all")
        messagebox.showinfo("Guardado", f"Imagen guardada como {filename}")
        self.load_images()



    def save_canvas_as_image(self, filename):
        """Guardar canvas con soporte para múltiples monitores"""
        try:
            # Obtener coordenadas relativas a la ventana principal
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            
            # Asegurar que el canvas esté visible antes de capturar
            self.canvas.update_idletasks()
            
            # Intentar capturar usando all_screens si está disponible (PIL más reciente)
            try:
                # Para PIL >= 9.0.0
                img = ImageGrab.grab(bbox=(x, y, x1, y1), all_screens=True)
            except:
                # Para versiones anteriores
                img = ImageGrab.grab(bbox=(x, y, x1, y1))
            
            # Verificar si la imagen está vacía (negra)
            if img.getextrema() == ((0, 0), (0, 0), (0, 0)):
                raise Exception("Captura vacía - probablemente fuera del monitor principal")
            
            img.save(filename, 'PNG')
            print(f"Imagen guardada: {filename}")
            return True
            
        except Exception as e:
            print(f"Error con ImageGrab: {e}")
            # Intentar método alternativo
            return self.save_canvas_as_image_alt(filename)
    
    def load_images(self):
        """Load existing images from saved_images directory using os.walk"""
        base_path = 'saved_images'
        self.image_list = []
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
                            self.image_list.append(file_path)
                            self.image_listbox.insert(tk.END, f"{file}")
                                

    def on_image_select(self, event):
        """Handle image selection from list"""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.image_list):
                image_path = self.image_list[index]
                self.display_image(image_path)

    def display_image(self, image_path):
        """Display selected image in the image display area"""
        try:
            img = Image.open(image_path)
            img.thumbnail((270, 270), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            self.image_display.delete("all")
            self.image_display.create_image(
                140, 140, image=photo, anchor=tk.CENTER)
            self.image_display.image = photo  # Keep a reference

            self.selected_image = image_path
        except Exception as e:
            messagebox.showerror(
                "Error", f"No se pudo cargar la imagen: {str(e)}")
  


if __name__ == "__main__":
    root = tk.Tk()
    app = CapturarNumeros(root)
    root.mainloop()
