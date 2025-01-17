from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import threading
from kivy.clock import Clock

from prediction import predict_audio  # Assuming `predict_audio` is in prediction.py

class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        
        self.label = Label(text="Scream Detection System", font_size=24, size_hint=(1, 0.2))
        self.add_widget(self.label)
        
        self.file_chooser = FileChooserIconView(size_hint=(1, 0.6))
        self.add_widget(self.file_chooser)
        
        self.detect_button = Button(text="Detect Scream", size_hint=(1, 0.2))
        self.detect_button.bind(on_press=self.detect_scream)
        self.add_widget(self.detect_button)
    
    def detect_scream(self, instance):
        selected_file = self.file_chooser.selection
        if not selected_file:
            self.show_popup("Error", "Please select an audio file!")
            return
        
        self.show_popup("Processing", "Analyzing audio...")
        
        # Run prediction in a thread to avoid UI freezing
        threading.Thread(target=self.run_prediction, args=(selected_file[0],)).start()
    
    def run_prediction(self, file_path):
     try:
        result = predict_audio(file_path)  # Call the prediction function
        # Schedule the UI update on the main thread
        Clock.schedule_once(lambda dt: self.show_popup("Result", result))
     except Exception as e:
        # Handle any exceptions and show an error popup
        Clock.schedule_once(lambda dt: self.show_popup("Error", str(e)))
    
    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(0.8, 0.5))
        popup.open()

class ScreamDetectionApp(App):
    def build(self):
        return MainScreen()

if __name__ == "__main__":
    ScreamDetectionApp().run()
