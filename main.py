import random
import customtkinter
import tkinter as tk
#Creating a GUI window with customertkinter and tkinter

app = customtkinter.CTk()
app.geometry("500x700")
app.title("Dice Roller")

label_sub = customtkinter.CTkLabel(master=app, text="Press the button to roll the die:", font=('Arial', 20))
label_sub.place(relx=0.5, rely=0.12, anchor=tk.CENTER)

Label = customtkinter.CTkLabel(app, text="", font=("Arial", 150))

def roll():
    number = random.randint(1,6)
    Label.configure(text=number)
    Label.pack(padx=20, pady=20)    
def button_callback():
    print("Die roll")

button = customtkinter.CTkButton(app, text="Roll Die...",font=('Arial', 13), text_color=("gray10", "#DCE4EE"),fg_color=("#DB3E39", "#821D1A"),hover_color="green",height=50, width=100,anchor="center", command=roll)
button.pack(padx=20, pady=20)

app.mainloop()



