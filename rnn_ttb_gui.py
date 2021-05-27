"""""""""""""""""""""""""""""""""""""""
rnn_ttb_gui.py
Модуль графического интерфейса генератора.
Группа: КЭ-120
ФИО:
    Глизница Максим Николаевич 
    Снегирева Дарья Алексеевна
"""""""""""""""""""""""""""""""""""""""
import tkinter as tk # Графический интерфейс
import rnn_ttb_load # Файл генератора предложений
import csv # Для сохранения данных в формате csv
import json # Для сохранения данных в формате json

class SWin(tk.Toplevel):
    def __init__(self, master = None):
        """
        Функция инициализации окна предложений.

        Parameters
        ----------
        master : tkinter.Tk, optional
            Родительский объект.
            
        """
        super().__init__(master = master)
        self.title('Root window')
        self.content = tk.Frame(self)
        self.lbl_stc = tk.Label(self.content, text = 'Press \'Next sentence\'')
        self.lbl_tmp = tk.Label(self.content, text = 'Temperature:')
        self.scl_tmp = tk.Scale(self.content, from_ = 0.1, to = 2, orient = tk.HORIZONTAL, resolution = 0.1)
        self.btn_good = tk.Button(self.content, text = 'Good sentence')
        self.btn_bad = tk.Button(self.content, text = 'Malformed sentence')
        self.btn_next = tk.Button(self.content, text = 'Next sentence')
        
        self.btn_good.bind('<Button-1>', self.good_sentence)
        self.btn_bad.bind('<Button-1>', self.bad_sentence)
        self.btn_next.bind('<Button-1>', self.next_sentence)
        
        self.content.grid(row = 0, column = 0)
        self.lbl_stc.grid(column = 0, row = 0, columnspan = 5, rowspan = 1)
        self.lbl_tmp.grid(column = 1, row = 1)
        self.scl_tmp.grid(column = 3, row = 1)
        self.btn_good.grid(column = 0, row = 2)
        self.btn_bad.grid(column = 2, row = 2)
        self.btn_next.grid(column = 4, row = 2)
        
        for col in range(self.content.grid_size()[0]):
            self.content.grid_columnconfigure(col, minsize = 120)
            
        for row in range(self.content.grid_size()[1]):
            self.content.grid_rowconfigure(row, minsize = 50)
        
    def good_sentence(self, event):
        """
        Сохранение предложения в файл с удачными примерами.

        Parameters
        ----------
        event : tkinter.Event
            Объект события.

        """
        print(type(event))
        with open('good.csv', 'a', newline = '') as handle:
            csv.writer(handle).writerow([self.lbl_stc['text'], 'present simple'])

    def bad_sentence(self, event):
        """
        Сохранение предложения в файл с неудачными примерами.

        Parameters
        ----------
        event : tkinter.Event
            Объект события.

        """
        with open('bad.csv', 'a', newline = '') as handle:
            csv.writer(handle).writerow([self.lbl_stc['text'], 'malformed'])
    
    def next_sentence(self, event):
        """
        Генерация следующего предложения.

        Parameters
        ----------
        event : tkinter.Event
            Объект события.

        """
        self.lbl_stc['text'] = rnn_ttb_load.generate_sentence(self.scl_tmp.get())

class EWin(tk.Toplevel):
    exc = []
    def __init__(self, master = None):
        """
        Функция инициализации окна упражнений.

        Parameters
        ----------
        master : tkinter.Tk, optional
            Родительский объект.
            
        """
        super().__init__(master = master)
        self.title('Root window')
        self.content = tk.Frame(self)
        self.lbl_stc = tk.Label(self.content, text = 'Press \'Next exercise\'')
        self.lbl_tmp = tk.Label(self.content, text = 'Temperature:')
        self.scl_tmp = tk.Scale(self.content, from_ = 0.1, to = 2, orient = tk.HORIZONTAL, resolution = 0.1)
        self.msg_a1 = tk.Message(self.content, text = 'Answers number 1')
        self.msg_a2 = tk.Message(self.content, text = 'Answers number 2')
        self.msg_a3 = tk.Message(self.content, text = 'Answers number 3')
        self.btn_save = tk.Button(self.content, text = 'Save exercise')
        self.btn_next = tk.Button(self.content, text = 'Next exercise')
        
        self.btn_save.bind('<Button-1>', self.save_exercise)
        self.btn_next.bind('<Button-1>', self.next_exercise)
        
        self.content.grid(row = 0, column = 0)
        self.lbl_stc.grid(column = 0, row = 0, columnspan = 5, rowspan = 1)
        self.lbl_tmp.grid(column = 0, row = 1)
        self.scl_tmp.grid(column = 1, row = 1)
        self.msg_a1.grid(column = 2, row = 1)
        self.msg_a2.grid(column = 3, row = 1)
        self.msg_a3.grid(column = 4, row = 1)
        self.btn_save.grid(column = 0, row = 2)
        self.btn_next.grid(column = 4, row = 2)
        
        for col in range(self.content.grid_size()[0]):
            self.content.grid_columnconfigure(col, minsize = 120)
            
        for row in range(self.content.grid_size()[1]):
            self.content.grid_rowconfigure(row, minsize = 50)
        
    def save_exercise(self, event):
        """
        Сохранение упражнения в файл.

        Parameters
        ----------
        event : tkinter.Event
            Объект события.

        """
        print(EWin.exc)
        try:
            with open('exc.json', 'r') as handle:
                excs = json.load(handle)
        except:
           excs = {'exercises': []}
        excs['exercises'].append({'example': EWin.exc[0], 'options': EWin.exc[1]})
        with open('exc.json', 'w') as handle:
            json.dump(excs, handle)
    
    def next_exercise(self, event):
        """
        Генерация следующего упражнения.

        Parameters
        ----------
        event : tkinter.Event
            Объект события.

        """
        try:
            #raise ValueError
            EWin.exc = rnn_ttb_load.generate_exercise(self.scl_tmp.get())
            print(EWin.exc)
            self.lbl_stc['text'] = EWin.exc[0]
            self.msg_a1['text'] = ''
            self.msg_a2['text'] = ''
            self.msg_a3['text'] = ''
            if(len(EWin.exc[1]) > 0):
                self.msg_a1['text'] = EWin.exc[1][0] + '\n' + EWin.exc[1][1][0] + '\n' + EWin.exc[1][1][1]
            if(len(EWin.exc[1]) > 2):
                self.msg_a2['text'] = EWin.exc[1][2] + '\n' + EWin.exc[1][3][0] + '\n' + EWin.exc[1][3][1]
            if(len(EWin.exc[1]) > 4):
                self.msg_a3['text'] = EWin.exc[1][4] + '\n' + EWin.exc[1][5][0] + '\n' + EWin.exc[1][5][1]
        except:
            self.lbl_stc['text'] = 'Internal error. Please try again.'
            EWin.exc = []
        pass

# Создание основного окна
root = tk.Tk()
content = tk.Frame(root, bg = 'white')
btn_sw = tk.Button(content, text = 'Sentences')
btn_ew = tk.Button(content, text = 'Exercises')
btn_sw.bind('<Button-1>', lambda e: SWin(root))
btn_ew.bind('<Button-1>', lambda e: EWin(root))

content.grid(row = 0, column = 0)
btn_sw.grid(column = 0, row = 2)
btn_ew.grid(column = 1, row = 2)

for col in range(content.grid_size()[0]):
    content.grid_columnconfigure(col, minsize = 120)
    
for row in range(content.grid_size()[1]):
    content.grid_rowconfigure(row, minsize = 50)

root.mainloop()