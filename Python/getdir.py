def getdir(request):
    import tkinter, os
    from tkinter import filedialog
    root = tkinter.Tk()
    path = filedialog.askdirectory(title = request,
           initialdir = os.getcwd(), parent = root)
    root.destroy()
    return path