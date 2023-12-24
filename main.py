import random
import customtkinter
#Creating a class for the GUI window with customertkinter

app = customtkinter.CTk()
app.geometry("700x700")

Label = customtkinter.CTkLabel(app, text="", font=("Arial", 300))

def roll():
    number = random.randint(1,6)
    Label.configure(text=number)
    Label.pack(padx=20, pady=20)    
def button_callback():
    print("Die roll")

button = customtkinter.CTkButton(app, text="Roll die...",fg_color=("#DB3E39", "#821D1A"),hover_color="green",height=50, width=100,anchor="center", command=roll)
button.pack(padx=20, pady=20)

app.mainloop()



