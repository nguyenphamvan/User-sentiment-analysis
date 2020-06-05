from flask import Flask, flash, render_template, redirect, url_for, request
import demo

model = demo.load_model()

app = Flask(__name__)



@app.route('/')
def hello_world():
    return redirect('/classify')

@app.route('/classify', methods=['GET', 'POST'])
def classify_system():
    error = None
    if request.method == 'POST':
        comment = request.form['comment']
        if comment == '':
            error = 'Bạn chưa nhập dữ liệu. Hãy nhập lại'
            return render_template('UI_demo.html', error=error)
        else:
            predict = demo.classify_one_comment(model,comment)[-1]
            print(predict)
            if predict == 0:
                emotional = "Tích cực"
            else:
                emotional = "Tiêu cực"
            return render_template('UI_demo.html', error=error, comment= comment, emotional=emotional)
    else:
        return render_template('UI_demo.html')



if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)

    # print(url_for('hello_world'))
