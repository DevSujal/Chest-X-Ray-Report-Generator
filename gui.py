import config
import utils
import numpy as np

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog


label = None
image = None
model = None

def choose_image():
    global label, image

    path = filedialog.askopenfilename(initialdir='images', title='Select Photo')

    screen = Toplevel(root)
    screen.title('Report Generator')

    ff1 = Frame(screen, bg='grey', borderwidth=6, relief=GROOVE)
    ff1.pack(side=TOP,fill=X)

    ff2 = Frame(screen, bg='grey', borderwidth=6, relief=GROOVE)
    ff2.pack(side=TOP, fill=X)

    ff4 = Frame(screen, bg='grey', borderwidth=6, relief=GROOVE)
    ff4.pack(side=TOP, fill=X)

    ff3 = Frame(screen, bg='grey', borderwidth=6, relief=GROOVE)
    ff3.pack(side=TOP, fill=X)

    Label(ff1, text='Select X-Ray', fg='white', bg='grey', font='Helvetica 16 bold').pack()

    original_img = Image.open(path).convert('L')

    image = np.array(original_img)
    image = np.expand_dims(image, axis=-1)
    image = image.repeat(3, axis=-1)

    image = config.basic_transforms(image=image)['image']

    photo = ImageTk.PhotoImage(original_img)

    Label(ff2, image=photo).pack()
    label = Label(ff4, text='', fg='blue', bg='gray', font='Helvetica 16 bold')
    label.pack()

    Button(ff3, text='Generate Report', bg='violet', command=generate_report, height=2, width=20, font='Helvetica 16 bold').pack(side=LEFT)
    Button(ff3, text='Quit', bg='red', command=quit_gui, height=2, width=20, font='Helvetica 16 bold').pack()

    screen.bind('<Configure>', lambda event: label.configure(wraplength=label.winfo_width()))
    screen.mainloop()

def generate_report():
    global label, image, model

    model.eval()

    image = image.to(config.DEVICE)

    report = model.generate_caption(image.unsqueeze(0), max_length=25)

    label.config(text=report, fg='violet', bg='green', font='Helvetica 16 bold', width=40)
    label.update_idletasks()

def quit_gui():
    root.destroy()

root = Tk()
root.title('Chest X-Ray Report Generator')

f1 = Frame(root, bg='grey', borderwidth=6, relief=GROOVE)
f1.pack(side=TOP, fill=X)

f2 = Frame(root, bg='grey', borderwidth=6, relief=GROOVE)
f2.pack(side=TOP, fill=X)

Label(f1, text='Welcome to Chest X-Ray Report Generator', fg='white', bg='grey', font='Helvetica 16 bold').pack()

btn1 = Button(root, text='Choose Chest X-Ray', command=choose_image, height=2, width=20, bg='blue', font="Helvetica 16 bold", pady=10)
btn1.pack()

Button(root, text='Quit', command=quit_gui, height=2, width=20, bg='violet', font='Helvetica 16 bold', pady=10).pack()

if __name__ == '__main__':
    model = utils.get_model_instance(utils.load_dataset().vocab)
    
    utils.load_checkpoint(model)

    root.mainloop()