import os
import numpy as np
import cv2 as cv
from tkinter import Tk, Label, Button, filedialog, Canvas, StringVar, OptionMenu, HORIZONTAL, VERTICAL, Scale, Frame, Toplevel, Text, Scrollbar, RIGHT, LEFT, Y, W, E, END
from PIL import Image, ImageTk
import pydicom
from  natsort import natsorted
class WatershedApp:
    def __init__(self, master):
        self.master = master
        master.title("Watershed Segmentation Tool")
        master.geometry("1200x800")
        master.configure(bg="#f0f0f0")

        self.main_frame = Frame(master, bg="#f0f0f0")
        self.main_frame.pack(expand=True, fill="both")

        self.top_frame = Frame(self.main_frame, bg="#f0f0f0")
        self.top_frame.pack(pady=10)

        self.label = Label(self.top_frame, text="Watershed Segmentation Tool", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        self.label.grid(row=0, columnspan=4, pady=10)

        self.upload_button = Button(self.top_frame, text="Upload DICOM Folder", command=self.upload_image, font=("Helvetica", 12), bg="#4CAF50", fg="white")
        self.upload_button.grid(row=1, column=0, pady=5, padx=5, sticky=W)

        self.readme_button = Button(self.top_frame, text="Read Me", command=self.show_readme, font=("Helvetica", 12), bg="#008CBA", fg="white")
        self.readme_button.grid(row=1, column=3, pady=5, padx=5, sticky=E)

        # Left column
        self.threshold_label = Label(self.top_frame, text="Threshold Value (0-255):", font=("Helvetica", 12), bg="#f0f0f0")
        self.threshold_label.grid(row=2, column=0, sticky='e', padx=10, pady=5)
        self.threshold_scale = Scale(self.top_frame, from_=0, to=255, orient=HORIZONTAL)
        self.threshold_scale.set(128)
        self.threshold_scale.grid(row=2, column=1, pady=5)

        self.morph_iter_label = Label(self.top_frame, text="Morphological Iterations:", font=("Helvetica", 12), bg="#f0f0f0")
        self.morph_iter_label.grid(row=3, column=0, sticky='e', padx=10, pady=5)
        self.morph_iter_scale = Scale(self.top_frame, from_=1, to=10, orient=HORIZONTAL)
        self.morph_iter_scale.set(2)
        self.morph_iter_scale.grid(row=3, column=1, pady=5)

        self.method_label = Label(self.top_frame, text="Threshold Method:", font=("Helvetica", 12), bg="#f0f0f0")
        self.method_label.grid(row=4, column=0, sticky='e', padx=10, pady=5)
        self.method_var = StringVar(self.top_frame)
        self.method_var.set("Otsu")
        self.method_options = ["Otsu", "Triangular", "Adaptive Mean", "Adaptive Gaussian"]
        self.method_menu = OptionMenu(self.top_frame, self.method_var, *self.method_options)
        self.method_menu.grid(row=4, column=1, pady=5)

        self.morph_label = Label(self.top_frame, text="Morphological Operation:", font=("Helvetica", 12), bg="#f0f0f0")
        self.morph_label.grid(row=5, column=0, sticky='e', padx=10, pady=5)
        self.morph_var = StringVar(self.top_frame)
        self.morph_var.set("Opening")
        self.morph_options = ["Opening", "Closing"]
        self.morph_menu = OptionMenu(self.top_frame, self.morph_var, *self.morph_options)
        self.morph_menu.grid(row=5, column=1, pady=5)

        # Right column
        self.dist_transform_label = Label(self.top_frame, text="Distance Transform Threshold (0-1):", font=("Helvetica", 12), bg="#f0f0f0")
        self.dist_transform_label.grid(row=2, column=2, sticky='e', padx=10, pady=5)
        self.dist_transform_scale = Scale(self.top_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
        self.dist_transform_scale.set(0.05)
        self.dist_transform_scale.grid(row=2, column=3, pady=5)

        self.distance_kernel_size_label = Label(self.top_frame, text="Distance Transform Kernel Size:", font=("Helvetica", 12), bg="#f0f0f0")
        self.distance_kernel_size_label.grid(row=3, column=2, sticky='e', padx=10, pady=5)
        self.distance_kernel_size_scale = Scale(self.top_frame, from_=1, to=10, orient=HORIZONTAL)
        self.distance_kernel_size_scale.set(5)
        self.distance_kernel_size_scale.grid(row=3, column=3, pady=5)

        self.kernel_size_label = Label(self.top_frame, text="Kernel Size:", font=("Helvetica", 12), bg="#f0f0f0")
        self.kernel_size_label.grid(row=4, column=2, sticky='e', padx=10, pady=5)
        self.kernel_size_scale = Scale(self.top_frame, from_=1, to=10, orient=HORIZONTAL)
        self.kernel_size_scale.set(3)
        self.kernel_size_scale.grid(row=4, column=3, pady=5)

        self.operation_label = Label(self.top_frame, text="Operation Type:", font=("Helvetica", 12), bg="#f0f0f0")
        self.operation_label.grid(row=5, column=2, sticky='e', padx=10, pady=5)
        self.operation_var = StringVar(self.top_frame)
        self.operation_var.set("Erosion")
        self.operation_options = ["Erosion", "Dilation"]
        self.operation_menu = OptionMenu(self.top_frame, self.operation_var, *self.operation_options)
        self.operation_menu.grid(row=5, column=3, pady=5)

        self.segment_button = Button(self.top_frame, text="Segment Image", command=self.segment_image, font=("Helvetica", 12), bg="#f44336", fg="white")
        self.segment_button.grid(row=6, columnspan=4, pady=10)

        self.middle_frame = Frame(self.main_frame, bg="#f0f0f0")
        self.middle_frame.pack()

        self.canvas = Canvas(self.middle_frame, width=1000, height=600, bg="white")
        self.canvas.pack(side=LEFT)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.bbox = None

        self.scrollbar = Scrollbar(self.middle_frame, orient=VERTICAL, command=self.scroll)
        self.scrollbar.pack(side=RIGHT, fill=Y)


        self.bottom_frame = Frame(self.main_frame, bg="#f0f0f0")
        self.bottom_frame.pack()

        self.image_info = Label(self.bottom_frame, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.image_info.pack()

        self.zoom_factor = 1.0

    def show_readme(self):
        readme_window = Toplevel(self.master)
        readme_window.title("Read Me")
        readme_window.geometry("800x600")

        scrollbar = Scrollbar(readme_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        readme_text = Text(readme_window, wrap="word", yscrollcommand=scrollbar.set, font=("Helvetica", 12))
        readme_text.pack(expand=True, fill="both")

        readme_content = """
       ### Watershed Segmentation Tool Kullanım Kılavuzu ###

        Bu araç, görüntü segmentasyon işlemlerini gerçekleştirmek için tasarlanmıştır. Aşağıda arayüzde bulunan her bir bileşenin açıklamaları ve segmentasyon işlemini gerçekleştirmek için adım adım talimatlar yer almaktadır.

        1. **Upload Image Button**: Kullanıcının yerel dosya sisteminden bir görüntü dosyası yüklemesini sağlar.
        2. **Threshold Value (0-255)**: İkili eşikleme işlemi için kullanılacak olan eşik değerini belirtir. Bu değer, 0 ile 255 arasında olmalıdır.
        3. **Morphological Iterations**: Morfolojik işlemlerin kaç kez tekrarlanacağını belirtir. Gürültüyü gidermek veya belirli yapısal değişiklikler yapmak için kullanılır.
        4. **Distance Transform Threshold (0-1)**: Mesafe dönüşümünün sonuçlarına uygulanacak eşik değeri oranını belirtir. Bu değer, 0 ile 1 arasında olmalıdır.
        5. **Distance Transform Kernel Size**: Mesafe dönüşümü işlemi sırasında kullanılacak olan kernel boyutunu belirtir. Örneğin, 3 değeri 3x3 kernel kullanacağını ifade eder.
        6. **Kernel Size**: Morfolojik işlemler sırasında kullanılacak olan kernel boyutunu belirtir. Örneğin, 3 değeri 3x3 kernel kullanacağını ifade eder.
        7. **Threshold Method**: Eşikleme yöntemini seçmek için kullanılır. Otsu, Triangular, Adaptive Mean ve Adaptive Gaussian seçeneklerini içerir.
        8. **Morphological Operation**: Uygulanacak morfolojik işlemi seçmek için kullanılır. Açma (Opening) veya kapama (Closing) işlemlerinden birini seçebilirsiniz.
        9. **Operation Type**: Morfolojik işlemin türünü belirtir. Erozyon (Erosion) veya genişletme (Dilation) seçeneklerini içerir.
        10. **Segment Image Button**: Belirtilen parametrelere göre görüntü segmentasyon işlemini başlatır.

        ### Adım Adım Talimatlar ###

        1. "Upload Image" butonuna tıklayarak bir görüntü dosyası yükleyin.
        2. Eşik değeri, morfolojik iterasyon sayısı ve mesafe dönüşüm eşiği gibi parametreleri belirleyin.
        3. Kullanmak istediğiniz eşikleme yöntemini, morfolojik işlemi ve işlem türünü seçin.
        4. Kernel boyutlarını ve mesafe dönüşüm kernel boyutunu belirtin.
        5. "Segment Image" butonuna tıklayarak segmentasyon işlemini başlatın.
        6. Seçtiğiniz parametrelere göre segmentasyon sonuçlarını görüntüleyin.

        ### Eşikleme Yöntemleri ###
        1. **Otsu**: Görüntünün histogramını kullanarak en uygun eşik değerini otomatik olarak belirler, genellikle ikili görüntülerin segmentasyonu için kullanılır.
        2. **Triangular**: Histogramın üçgen yöntemini kullanarak eşik değeri belirler, unimodal histogramlar için uygundur.
        3. **Adaptive Mean**: Yerel bölgedeki piksellerin ortalamasını kullanarak eşik değeri belirler, aydınlatma değişikliklerinin olduğu görüntülerde etkilidir.
        4. **Adaptive Gaussian**: Yerel bölgedeki piksellerin ağırlıklı ortalamasını (Gaussian) kullanarak eşik değeri belirler, daha düzgün sonuçlar için kullanılır.

        ### Morfolojik İşlemler ###
        1. **Açma (Opening)**: Gürültüyü gidermek ve küçük nesneleri yok etmek için kullanılır; önce erozyon, sonra genişletme işlemlerini içerir.
        2. **Kapama (Closing)**: Küçük delikleri doldurmak ve nesnelerin sınırlarını düzleştirmek için kullanılır; önce genişletme, sonra erozyon işlemlerini içerir.
        3. **Erozyon (Erosion)**: Nesneleri küçültmek ve ince detayları ortadan kaldırmak için kullanılır; genellikle gürültü azaltma amaçlıdır.
        4. **Genişletme (Dilation)**: Nesneleri büyütmek ve küçük boşlukları doldurmak için kullanılır; nesnelerin sınırlarını genişletir.

        Bu bilgiler doğrultusunda segmentasyon işlemlerinizi gerçekleştirebilirsiniz. Herhangi bir sorunuz veya yardım ihtiyacınız olursa, bu kılavuzu tekrar inceleyebilirsiniz.
        """

        readme_text.insert(END, readme_content)
        readme_text.config(state="disabled")
        scrollbar.config(command=readme_text.yview)

    def upload_image(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            print("No folder selected.")
            return

        self.dicom_series = []
        for filename in natsorted(os.listdir(folder_path)):
            if filename.endswith(".dcm"):
                self.dicom_series.append(os.path.join(folder_path, filename))

        if self.dicom_series:
            self.current_index = 0
            self.show_dicom_image(self.current_index)
            self.scrollbar.config(command=self.scroll)
            self.scrollbar.set(0, 1 / len(self.dicom_series))
            self.update_image_info()

    def show_dicom_image(self, index):
        self.dicom_data = pydicom.dcmread(self.dicom_series[index])
        self.img = self.dicom_data.pixel_array
        if self.img is not None:
            self.img = cv.normalize(self.img, None, 0, 255, cv.NORM_MINMAX)
            self.img = np.uint8(self.img)
            self.img = cv.resize(self.img, (int(1000 * self.zoom_factor), int(600 * self.zoom_factor)))
            self.show_image(self.img)

    def show_image(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.update_image_info()

    def update_image_info(self):
        self.image_info.config(text=f"Image {self.current_index + 1}/{len(self.dicom_series)}")

    def on_canvas_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = None

    def on_mouse_drag(self, event):
        if self.start_x and self.start_y:
            if self.rect_id is not None:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='blue', width=3)

    def on_mouse_release(self, event):
        self.bbox = (self.start_x, self.start_y, event.x, event.y)
        print(f"Bounding box: {self.bbox}")

    def segment_image(self):
        if not hasattr(self, 'img'):
            print("No image uploaded.")
            return
        if not self.bbox:
            print("No bounding box selected.")
            return

        x1, y1, x2, y2 = self.bbox
        roi = self.img[y1:y2, x1:x2]

        threshold_value = self.threshold_scale.get()
        morph_iterations = self.morph_iter_scale.get()
        dist_transform_threshold = self.dist_transform_scale.get()
        kernel_size = self.kernel_size_scale.get()
        distance_kernel_size = self.distance_kernel_size_scale.get()

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Ensure the image is grayscale
        if len(roi.shape) == 3:
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        else:
            gray = roi

        method = self.method_var.get()
        if method == "Otsu":
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        elif method == "Triangular":
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
        elif method == "Adaptive Mean":
            thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
        elif method == "Adaptive Gaussian":
            thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

        morph_operation = self.morph_var.get()
        morph_type = self.operation_var.get()
        if morph_operation == "Opening":
            if morph_type == "Erosion":
                morph_result = cv.erode(thresh, kernel, iterations=morph_iterations)
            elif morph_type == "Dilation":
                morph_result = cv.dilate(thresh, kernel, iterations=morph_iterations)
        elif morph_operation == "Closing":
            if morph_type == "Erosion":
                morph_result = cv.dilate(thresh, kernel, iterations=morph_iterations)
                morph_result = cv.erode(morph_result, kernel, iterations=morph_iterations)
            elif morph_type == "Dilation":
                morph_result = cv.erode(thresh, kernel, iterations=morph_iterations)
                morph_result = cv.dilate(morph_result, kernel, iterations=morph_iterations)

        sure_bg = cv.dilate(morph_result, kernel, iterations=3)

        dist_transform = cv.distanceTransform(morph_result, cv.DIST_L2, distance_kernel_size)
        ret, sure_fg = cv.threshold(dist_transform, dist_transform_threshold * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        roi_color = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(roi_color, markers)
        roi_color[markers == -1] = [0, 0, 255]  # Change color to red

        self.img[y1:y2, x1:x2] = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)  # Convert back to grayscale if needed

        self.show_image(self.img)

    def scroll(self, *args):
        if args[0] == "scroll":
            self.current_index += int(args[1])
        elif args[0] == "moveto":
            fraction = float(args[1])
            self.current_index = int(fraction * (len(self.dicom_series) - 1))

        if self.current_index < 0:
            self.current_index = 0
        elif self.current_index >= len(self.dicom_series):
            self.current_index = len(self.dicom_series) - 1

        self.show_dicom_image(self.current_index)

    def zoom(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1
        elif event.delta < 0:
            self.zoom_factor /= 1.1

        self.show_dicom_image(self.current_index)

root = Tk()
app = WatershedApp(root)
root.mainloop()
