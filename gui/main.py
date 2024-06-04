import PySimpleGUIQt as sg
from retreiver import retrieve_from_baike
gecko_path = '/home/yueyulin/firefox/geckodriver'
firefox_path = '/home/yueyulin/firefox/firefox'

# All the stuff inside your window.
layout = [ [sg.Text('请输入问题：'), sg.InputText(key='输入问题')],
           [sg.Button('提交'), sg.Button('清空内容')] ,
           [sg.Multiline(key='-TABLE-',size=(80,20),visible=False)]
        ]

# Create the Window
window = sg.Window('AI提问助手', layout,size=(640, 480))
# Finalize the window and get the screen size
window.finalize()
screen_width, screen_height = window.get_screen_dimensions()

# Calculate the position to center the window
x = (screen_width - window.size[0]) // 2
y = (screen_height - window.size[1]) // 2

# Move the window to the center of the screen
window.move(x, y)
# Event Loop to process "events" and get the "values" of the inputs
def group_texts(texts,max_length=512):
    chunks = []
    current_chunk = ''
    for text in texts:
        text = text.strip()
        if len(current_chunk) + len(text) > max_length:
            chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk += '\n' + text
    chunks.append(current_chunk)
    chunks = [chunk.replace('\n','').replace('\t','') for chunk in chunks]
    return chunks
while True:
    event, values = window.read()

    # if user closes window or clicks cancel
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break

    if event == '清空内容':
        window['输入问题'].update('')
    if event == '提交':
        query = values['输入问题']
        print('You entered ', query)
        results = retrieve_from_baike(query,gecko_path,firefox_path)
        results = [(result,1.0) for result in group_texts(results)]
        print(len(results))
        window['-TABLE-'].update('搜索结果\n',text_color_for_value='blue',visible=True,append=False)
        for i in range(len(results)):
            window['-TABLE-'].update(f'{i+1}:{results[i][0]}\n',append=True)
window.close()
