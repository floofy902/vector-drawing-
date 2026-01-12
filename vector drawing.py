import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- ADDED ---
# Try to import winsound for audio. This is Windows-only.
try:
    import winsound
    SOUND_ENABLED = True
except ImportError:
    SOUND_ENABLED = False
    print("Warning: 'winsound' module not found. Sound will be disabled. (This is expected on non-Windows systems)")
# --- END ADDED ---


class VectorDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Vector Drawing (Upgraded)")
        self.root.geometry("1600x900") # Increased size for new controls

        # --- Constants ---
        self.IMG_SIZE = (700, 700) # Increased image size

        # --- Application State Variables ---
        self.input_img = None
        self.vector_img = None
        self.edges_mask_bgr = None # To store the preview
        self.contours = []
        self.drawing = False
        self.current_contour = 0
        self.current_point = 0
        self.animation_id = None

        # --- Mode & Style Variables ---
        self.bg_color = (0, 0, 0) # Black
        self.line_color = (0, 255, 0) # Green
        self.use_color_sampling = BooleanVar()
        self.show_edge_preview = BooleanVar()
        self.sound_enabled_var = BooleanVar(value=True) # --- ADDED ---

        # --- UI Setup ---
        self.setup_ui()
        self.update_status("Welcome! Please load an image to begin.")

    def setup_ui(self):
        # --- Top Control Frame ---
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # --- Button Sub-Frame ---
        button_frame = Frame(control_frame)
        button_frame.pack(pady=5)

        Button(button_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        
        self.start_pause_button = Button(button_frame, text="Start Drawing", command=self.start_drawing, state=DISABLED)
        self.start_pause_button.grid(row=0, column=1, padx=5)

        self.reset_button = Button(button_frame, text="Reset", command=self.reset_drawing, state=DISABLED)
        self.reset_button.grid(row=0, column=2, padx=5)

        self.finish_now_button = Button(button_frame, text="Finish Now", command=self.finish_now, state=DISABLED)
        self.finish_now_button.grid(row=0, column=3, padx=5)
        
        self.save_button = Button(button_frame, text="Save Output", command=self.save_image, state=DISABLED)
        self.save_button.grid(row=0, column=4, padx=5)

        # --- Slider Sub-Frame (NOW ENTRY WIDGETS) ---
        slider_frame = Frame(control_frame)
        slider_frame.pack(pady=5)

        Label(slider_frame, text="Canny Low:").grid(row=0, column=0, sticky=E, padx=(5,0))
        self.canny_low = Entry(slider_frame, width=8) # MODIFIED
        self.canny_low.insert(0, "10") # MODIFIED
        self.canny_low.grid(row=0, column=1, sticky=W, padx=(2,10))

        Label(slider_frame, text="Canny High:").grid(row=0, column=2, sticky=E, padx=(5,0))
        self.canny_high = Entry(slider_frame, width=8) # MODIFIED
        self.canny_high.insert(0, "60") # MODIFIED
        self.canny_high.grid(row=0, column=3, sticky=W, padx=(2,10))

        Label(slider_frame, text="Min Contour:").grid(row=0, column=4, sticky=E, padx=(5,0))
        self.min_contour_len = Entry(slider_frame, width=8) # MODIFIED
        self.min_contour_len.insert(0, "10") # MODIFIED
        self.min_contour_len.grid(row=0, column=5, sticky=W, padx=(2,10))

        Label(slider_frame, text="Line Thick:").grid(row=0, column=6, sticky=E, padx=(5,0))
        self.line_thickness = Entry(slider_frame, width=8) # MODIFIED
        self.line_thickness.insert(0, "1") # MODIFIED
        self.line_thickness.grid(row=0, column=7, sticky=W, padx=(2,10))

        Label(slider_frame, text="Speed:").grid(row=0, column=8, sticky=E, padx=(5,0))
        self.speed_scale = Entry(slider_frame, width=8) # MODIFIED
        self.speed_scale.insert(0, "500") # MODIFIED
        self.speed_scale.grid(row=0, column=9, sticky=W, padx=(2,10))

        # --- Mode Sub-Frame ---
        mode_frame = Frame(control_frame)
        mode_frame.pack(pady=5)

        self.reprocess_button = Button(mode_frame, text="Reprocess Edges", command=self.process_edges, state=DISABLED)
        self.reprocess_button.grid(row=0, column=0, padx=10)

        self.bg_toggle_button = Button(mode_frame, text="Toggle BG (B/W)", command=self.toggle_background, state=DISABLED)
        self.bg_toggle_button.grid(row=0, column=1, padx=10)
        
        Checkbutton(mode_frame, text="Use Sampled Color", variable=self.use_color_sampling).grid(row=0, column=2, padx=10)
        
        self.edge_preview_check = Checkbutton(mode_frame, text="Show Edge Preview", variable=self.show_edge_preview, command=self.update_vector_display)
        self.edge_preview_check.grid(row=0, column=3, padx=10)

        # --- ADDED ---
        self.sound_check = Checkbutton(mode_frame, text="Enable Sound", variable=self.sound_enabled_var)
        self.sound_check.grid(row=0, column=4, padx=10)
        
        # Disable sound checkbox if module failed to import
        if not SOUND_ENABLED:
            self.sound_enabled_var.set(False)
            self.sound_check.config(state=DISABLED, text="Sound (Windows Only)")
        # --- END ADDED ---

        # --- Image Display Frame ---
        image_frame = Frame(self.root)
        image_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        self.original_label = Label(image_frame, text="Original Image", bg="gray90")
        self.original_label.pack(side=LEFT, padx=10, fill=BOTH, expand=True)

        self.vector_label = Label(image_frame, text="Vector Output", bg="gray10")
        self.vector_label.pack(side=RIGHT, padx=10, fill=BOTH, expand=True)

        # --- Status Bar ---
        self.status_label = Label(self.root, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status_label.pack(side=BOTTOM, fill=X)

        # Create placeholder images
        self.reset_drawing() # Initialize vector_img
        placeholder = np.full((self.IMG_SIZE[1], self.IMG_SIZE[0], 3), 230, dtype=np.uint8) # Light gray
        self.display_image(placeholder, self.original_label)
        self.display_image(self.vector_img, self.vector_label)

    # --- NEW HELPER FUNCTION ---
    def get_int_from_entry(self, entry_widget, default_value, min_val=None):
        """Safely gets an integer from an Entry widget, with optional min clamping."""
        try:
            val = int(entry_widget.get())
            
            if min_val is not None and val < min_val:
                val = min_val
                # Update entry widget if value was clamped
                entry_widget.delete(0, END)
                entry_widget.insert(0, str(val))
                
            return val
            
        except ValueError:
            # If conversion fails, reset to default and return it
            entry_widget.delete(0, END)
            entry_widget.insert(0, str(default_value))
            return default_value
    # --- END NEW HELPER FUNCTION ---

    def update_status(self, message):
        """Updates the text in the bottom status bar."""
        self.status_label.config(text=message)

    def load_image(self):
        self.stop_drawing()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image file. It may be corrupt.")
            return

        self.input_img = cv2.resize(img, self.IMG_SIZE)
        self.update_status(f"Loaded {path.split('/')[-1]}")

        self.process_edges()

        self.display_image(self.input_img, self.original_label)

        # Enable buttons
        self.start_pause_button.config(state=NORMAL)
        self.save_button.config(state=NORMAL)
        self.reset_button.config(state=NORMAL)
        self.reprocess_button.config(state=NORMAL)
        self.finish_now_button.config(state=NORMAL)
        self.bg_toggle_button.config(state=NORMAL)

    def process_edges(self):
        """Finds contours and (if checked) displays the edge preview."""
        if self.input_img is None:
            return
            
        self.reset_drawing()

        gray = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)

        # MODIFIED: Get values from Entry widgets safely
        low_thresh = self.get_int_from_entry(self.canny_low, 10, min_val=0)
        high_thresh = self.get_int_from_entry(self.canny_high, 60, min_val=0)
        min_len = self.get_int_from_entry(self.min_contour_len, 10, min_val=2)

        edges = cv2.Canny(gray, low_thresh, high_thresh)
        edges = cv2.dilate(edges, None)
        self.edges_mask_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Filter contours by length
        filtered_cnts = [c for c in cnts if len(c) >= min_len]
        
        self.contours = sorted(filtered_cnts, key=cv2.contourArea, reverse=True)
        
        self.update_status(f"Found {len(self.contours)} contours (min length {min_len}).")
        self.update_vector_display()

    def update_vector_display(self):
        """Shows either the edge preview or the vector canvas."""
        if self.show_edge_preview.get() and self.edges_mask_bgr is not None:
            self.display_image(self.edges_mask_bgr, self.vector_label)
        else:
            self.display_image(self.vector_img, self.vector_label)

    def toggle_background(self):
        """Toggles the background and line color, then resets."""
        if self.bg_color == (0, 0, 0): # If black
            self.bg_color = (255, 255, 255) # To white
            self.line_color = (0, 0, 0) # Line to black
        else: # If white
            self.bg_color = (0, 0, 0) # To black
            self.line_color = (0, 255, 0) # Line to green
        
        self.update_status(f"Background set to {'white' if self.bg_color == (255,255,255) else 'black'}.")
        self.process_edges() # Re-process and reset

    def start_drawing(self):
        if not self.contours:
            messagebox.showwarning("No Contours", "No contours found. Try adjusting Canny values and 'Reprocess Edges'.")
            return
            
        if self.drawing:
            return

        # If preview is on, turn it off to show the drawing
        if self.show_edge_preview.get():
            self.show_edge_preview.set(False)
            self.update_vector_display()

        self.drawing = True
        self.start_pause_button.config(text="Pause", command=self.pause_drawing)
        self.reset_button.config(state=DISABLED)
        self.reprocess_button.config(state=DISABLED)
        self.finish_now_button.config(state=NORMAL)
        
        self.draw_step()

    def pause_drawing(self):
        self.drawing = False
        self.stop_drawing() # Stops the 'after' loop
        self.start_pause_button.config(text="Resume", command=self.start_drawing)
        self.reset_button.config(state=NORMAL)
        self.reprocess_button.config(state=NORMAL)
        self.update_status(f"Paused at contour {self.current_contour} of {len(self.contours)}.")

    def stop_drawing(self):
        """Stops the 'after' loop completely."""
        self.drawing = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

    def reset_drawing(self):
        """Stops drawing and clears the vector canvas to the BG color."""
        self.stop_drawing()
        
        self.current_contour = 0
        self.current_point = 0
        
        if self.input_img is not None:
            self.vector_img = np.full_like(self.input_img, self.bg_color, dtype=np.uint8)
        else:
            self.vector_img = np.full((self.IMG_SIZE[1], self.IMG_SIZE[0], 3), self.bg_color, dtype=np.uint8)

        self.display_image(self.vector_img, self.vector_label)
        
        self.start_pause_button.config(text="Start Drawing", command=self.start_drawing)
        if self.input_img is None:
            self.start_pause_button.config(state=DISABLED)
            self.reset_button.config(state=DISABLED)
            self.finish_now_button.config(state=DISABLED)
        else:
             self.start_pause_button.config(state=NORMAL)
             self.reset_button.config(state=NORMAL)
             self.finish_now_button.config(state=NORMAL)
        
        if self.contours:
             self.update_status(f"Canvas reset. {len(self.contours)} contours ready.")
        else:
             self.update_status("Canvas reset.")

    def finish_now(self):
        """Instantly draws all remaining contours."""
        if not self.contours:
            return
            
        self.stop_drawing()
        self.update_status("Finishing remaining drawing...")

        # Get settings *once*
        # MODIFIED: Get values from Entry widgets safely
        thickness = self.get_int_from_entry(self.line_thickness, 1, min_val=1)
        use_sampling = self.use_color_sampling.get()
        default_color = self.line_color
        
        # Loop from the current contour to the end
        for i in range(self.current_contour, len(self.contours)):
            pts = self.contours[i]
            
            # Start from current_point if it's the current contour, else start from 0
            start_pt = self.current_point if i == self.current_contour else 0
            
            for j in range(start_pt, len(pts) - 1):
                x1, y1 = pts[j][0]
                x2, y2 = pts[j + 1][0]
                
                if use_sampling:
                    # Sample color from midpoint
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    color = self.input_img[my, mx]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                else:
                    color = default_color
                
                cv2.line(self.vector_img, (x1, y1), (x2, y2), color, thickness)
        
        # Mark as done
        self.current_contour = len(self.contours)
        self.current_point = 0
        
        self.display_image(self.vector_img, self.vector_label)
        self.pause_drawing()
        self.start_pause_button.config(state=DISABLED)
        self.finish_now_button.config(state=DISABLED)
        self.update_status("Drawing complete.")

    # --- ADDED ---
    def map_y_to_freq(self, y):
        """Maps a y-coordinate (0=top) to a sound frequency."""
        min_freq = 300  # A low-ish note
        max_freq = 2000 # A high-ish note
        img_height = self.IMG_SIZE[1]
        
        # Normalize y from 0 (top) to 1 (bottom)
        percent_y = max(0, min(1, y / img_height))
        
        # Invert so 0 (top) = max_freq and 1 (bottom) = min_freq
        freq_range = max_freq - min_freq
        frequency = max_freq - (percent_y * freq_range)
        
        return int(frequency)
    # --- END ADDED ---

    def draw_step(self):
        if not self.drawing:
            return

        if self.current_contour >= len(self.contours):
            print("Drawing complete.")
            self.pause_drawing() 
            self.start_pause_button.config(state=DISABLED)
            self.finish_now_button.config(state=DISABLED)
            self.update_status("Drawing complete.")
            return

        # --- MODIFIED ---
        # Sound logic: Play a short beep based on the *current* Y position
        # This runs once per frame (i.e., once per call to draw_step)
        if SOUND_ENABLED and self.sound_enabled_var.get():
            try:
                # Get the y-coord of the *next* point to be drawn
                pts = self.contours[self.current_contour]
                if len(pts) > 0 and self.current_point < len(pts):
                    y_coord = pts[self.current_point][0][1]
                    freq = self.map_y_to_freq(y_coord)
                    # This is a *blocking* call, so duration must be short.
                    # 15ms will slightly slow the max framerate, creating the sound.
                    winsound.Beep(freq, 15) 
            except Exception as e:
                pass # Ignore sound errors (e.g., index out of bounds)
        # --- END MODIFIED ---

        # Get settings from sliders
        # MODIFIED: Get values from Entry widgets safely
        steps_per_frame = self.get_int_from_entry(self.speed_scale, 500, min_val=1)
        thickness = self.get_int_from_entry(self.line_thickness, 1, min_val=1)
        use_sampling = self.use_color_sampling.get()
        default_color = self.line_color

        # Update status bar only when the contour changes
        contour_changed = False

        for _ in range(steps_per_frame):
            if self.current_contour >= len(self.contours):
                break 

            pts = self.contours[self.current_contour]

            # We already pre-filtered, but good to keep a basic check
            if len(pts) < 2:
                self.current_contour += 1
                self.current_point = 0
                contour_changed = True
                continue 

            if self.current_point >= len(pts) - 1:
                self.current_contour += 1
                self.current_point = 0
                contour_changed = True
                continue 

            # Get points for the line segment
            x1, y1 = pts[self.current_point][0]
            x2, y2 = pts[self.current_point + 1][0]

            # --- Main Drawing Logic ---
            if use_sampling:
                # Sample color from midpoint (y, x for numpy)
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                color = self.input_img[my, mx]
                # Convert numpy.uint8 to Python int tuple
                color = (int(color[0]), int(color[1]), int(color[2]))
            else:
                color = default_color

            cv2.line(self.vector_img, (x1, y1), (x2, y2), color, thickness)
            self.current_point += 1
        
        if contour_changed and len(self.contours) > 0 and self.current_contour < len(self.contours):
            percent_done = (self.current_contour / len(self.contours)) * 100
            self.update_status(f"Drawing... {percent_done:.0f}% (Contour {self.current_contour}/{len(self.contours)})")

        self.display_image(self.vector_img, self.vector_label)
        self.animation_id = self.root.after(1, self.draw_step)

    def save_image(self):
        if self.vector_img is None:
            messagebox.showwarning("No Image", "There is no image to save.")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if path:
            try:
                cv2.imwrite(path, self.vector_img)
                self.update_status(f"Image saved to {path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving image: {e}")

    def display_image(self, img, widget):
        """Converts a cv2 (BGR) image to a Tkinter-compatible image and updates the widget."""
        
        # Handle 2D grayscale images (like the edge mask)
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        pil = Image.fromarray(rgb)
        
        # Resize to fit the label, which is better than fixed size
        pil = pil.resize(self.IMG_SIZE, Image.LANCZOS) 
        
        tk = ImageTk.PhotoImage(pil)
        
        widget.config(image=tk)
        widget.image = tk # Keep a reference


if __name__ == "__main__":
    root = Tk()
    app = VectorDrawingApp(root)
    root.mainloop()
