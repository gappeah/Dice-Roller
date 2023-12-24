import random
import customtkinter
#Creating a class for the GUI window with customertkinter
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("750x300")

        self.button = customtkinter.CTkButton(self, text="Roll the dice", command=self.button_callbck)
        self.button.pack(padx=20, pady=20)
    
        self.label = customtkinter.CTkLabel(App, text="", font=("time", 50))
    
    
    def roll(self):
        number =['\2680', '\2681', '\2682', '\2683', '\2684', '\2685']
        return random.randint(1, 6)
        

    def button_callbck(self):
        print("Dice rolled")

app = App()
app.mainloop()
