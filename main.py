import tkinter as tk
from tkinter import Canvas, Scrollbar, filedialog, messagebox, Label, Text, Frame, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
from scipy.spatial import KDTree
import pandas as pd

# Global variables for window size
window_width = 600
window_height = 600

# Read color data from CSV
try:
    # Load color data from CSV
    color_data = pd.read_csv('colors.csv', header=None,
                             names=["identifier", "color_name", "hex", "B", "G", "R"])

    # Create a dictionary with RGB tuples as keys and color names as values
    COLOR_NAMES = dict(zip(
        color_data[['B', 'G', 'R']].astype(int).apply(tuple, axis=1),
        color_data['color_name']
    ))
except Exception as e:
    print(f"Error loading color data: {e}")
    COLOR_NAMES = {}

color_values = np.array(list(COLOR_NAMES.keys()))
color_tree = KDTree(color_values)


def get_color_name(rgb_tuple):
    """Return the name of the closest color."""
    if not COLOR_NAMES:
        return "Unknown"
    # Convert RGB tuple to BGR for query
    bgr_tuple = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])
    _, idx = color_tree.query(bgr_tuple)
    return COLOR_NAMES[tuple(color_values[idx])]


def get_shape_name(num_sides, contour):
    """Return the shape name based on the number of sides."""
    x, y, w, h = cv2.boundingRect(contour)

    if num_sides >= 8:
        # If the number of sides is 8 or more, check for circle or oval
        _, radius = cv2.minEnclosingCircle(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio <= 1.1:
            return "Circle"
        else:
            return "Oval"
    elif num_sides == 3:
        return "Triangle"
    elif num_sides == 4:
        return "Quadrilateral"
    elif num_sides == 5:
        return "Pentagon"
    elif num_sides == 6:
        return "Hexagon"
    elif num_sides == 7:
        return "Heptagon"
    else:
        return "Polygon"


def select_image():
    """Select and process an image, then update UI with results."""
    global window_width, window_height

    path = filedialog.askopenfilename()
    if path:
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        _, threshold = cv2.threshold(
            blur, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(
            threshold, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100
        filtered_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        image_with_contours = image.copy()
        object_data = []
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_contours, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_contours, str(i + 1), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            area = cv2.contourArea(contour)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            colors = masked_image.reshape(-1, 3)
            colors = [tuple(color) for color in colors if not np.array_equal(
                color, [0, 0, 0])]
            if colors:
                most_common_color = Counter(colors).most_common(1)[0][0]
                color_name = get_color_name(most_common_color)
            else:
                color_name = "Unknown"
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour, 0.04 * perimeter, True)
            num_sides = len(approx)
            shape_name = get_shape_name(num_sides, contour)
            object_data.append({
                'id': i + 1,
                'color': color_name,
                'area': area,
                'shape': shape_name
            })

        # Resize images for display
        resize_width = 400
        resize_height = int(image.shape[0] * resize_width / image.shape[1])
        image_resized = cv2.resize(image, (resize_width, resize_height))
        image_with_contours_resized = cv2.resize(
            image_with_contours, (resize_width, resize_height))

        # Convert images from BGR to RGB for Tkinter display
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_with_contours_rgb = cv2.cvtColor(
            image_with_contours_resized, cv2.COLOR_BGR2RGB)

        # Create PhotoImage objects for Tkinter
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        image_with_contours_tk = ImageTk.PhotoImage(
            image=Image.fromarray(image_with_contours_rgb))

        # Update Tkinter labels with images
        panelA.configure(image=image_tk)
        panelA.image = image_tk
        panelB.configure(image=image_with_contours_tk)
        panelB.image = image_with_contours_tk

        # Update object details text widget
        entry_objects.delete(1.0, tk.END)
        object_count_label.configure(
            text=f"Total Objects: {len(object_data)}")
        for obj in object_data:
            entry_objects.insert(tk.END, f'Object {obj["id"]} - Color: {obj["color"]}, Area: {obj["area"]}, Shape: {obj["shape"]}\n')


def save_results():
    """Save the object data and image with contours to a file."""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "w") as file:
            file.write(
                f"Total Objects: {object_count_label.cget('text')}\n\n")
            file.write(entry_objects.get(1.0, tk.END))
        messagebox.showinfo("Saved", f"Results saved to {file_path}")


def reset_application():
    """Reset the application to its initial state."""
    global window_width, window_height
    window_width = 600
    window_height = 600
    window.geometry(f"{window_width}x{window_height}")
    panelA.configure(image='')
    panelB.configure(image='')
    object_count_label.configure(text="Total Objects: 0")
    entry_objects.delete(1.0, tk.END)


# Create the main window
window = tk.Tk()
window.title("Enhanced Object Counter")

# Set the initial size of the window
window.geometry(f"{window_width}x{window_height}")

# Set the background color for the main window
window.configure(bg='light gray')

# Create a Canvas widget with Scrollbars
canvas = Canvas(window, bg='light gray')
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create vertical and horizontal Scrollbars
scrollbar_y = Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = Scrollbar(window, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

# Configure the Canvas to use the Scrollbars
canvas.configure(yscrollcommand=scrollbar_y.set)
canvas.configure(xscrollcommand=scrollbar_x.set)

# Create a frame inside the Canvas to contain widgets
frame = Frame(canvas, bg='light gray')
canvas.create_window((0, 0), window=frame, anchor=tk.NW)

# Bind the Configure event to adjust the scroll region
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind('<Configure>', on_frame_configure)

# Create labels for displaying images
panelA = Label(frame, bg='light gray')
panelA.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

panelB = Label(frame, bg='light gray')
panelB.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

# Text widget to display object details
entry_objects = Text(frame, width=60, height=20, bg='white', fg='black', font=('Times New Roman', 12))
entry_objects.grid(row=1, column=0, columnspan=2, pady=10, sticky='nsew')

# Label for object count
object_count_label = Label(
    frame, text="Total Objects: 0", bg='light gray', fg='black', font=('Times New Roman', 14, 'bold'))
object_count_label.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')

# Buttons
button_select = Button(
    frame, text="Select Image", width=20, relief=tk.SUNKEN, command=select_image, bg='light blue', fg='black', font=('Times New Roman', 12))
button_select.grid(row=3, column=0, pady=10, sticky='nsew')

button_save = Button(
    frame, text="Save Results", width=20, relief=tk.SUNKEN, command=save_results, bg='light green', fg='black', font=('Times New Roman', 12))
button_save.grid(row=3, column=1, pady=10, sticky='nsew')

button_reset = Button(
    frame, text="Reset", width=20, relief=tk.SUNKEN, command=reset_application, bg='light coral', fg='black', font=('Times New Roman', 12))
button_reset.grid(row=4, column=0, columnspan=2, pady=10, sticky='nsew')

# Start the Tkinter main loop
window.mainloop()

