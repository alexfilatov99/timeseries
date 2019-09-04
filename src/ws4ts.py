#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from fbprophet import Prophet
import logging
from flask import Flask, request, abort
import sys






# инициализация логгирования

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)






def fb_predict(data, n_period, freq):

# функция получает на вход временной ряд и возвращает верхнюю границу предсказанного ряда на заданное количество периодов вперед в формате json

# ----------
# Параметры:
# data - временной ряд в виде матрицы со столбцами 'ds' и 'y', где
#       ds - времнные индексы в текстовом формате
#       у - значения
# n_period - кол-во периодов предсказания, целое число
# freq - размерность периодов: 'D', 'W', 'Q', 'Y'

    try:
        print('freq=', freq)
        print('n_period=', n_period)
        logging.info('1.Читаю данные')
        data = pd.read_json(data)
        data['ds'] = pd.to_datetime(data['ds'], yearfirst=True)
        new_index = pd.date_range(data['ds'].min(), data['ds'].max(), freq=freq)
        #print(data)
        #print(new_index)
        data = data.sort_index()
        #print(data)
        logging.info('2.Данные прочитаны. Инициализация модели')
        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        logging.info('3.Модель инициализирована. Начинаю обучение модели')
        model.fit(data)
        logging.info('4.Обучение модели закончено')
        future = model.make_future_dataframe(periods=n_period, freq=freq)
        logging.info('5.Делаю предсказания')
        preds = model.predict(future)[-n_period:]
        logging.info('6.Предсказания сделаны. Формирую результат')
        preds['yhat_upper'] = round(preds['yhat_upper'])
        logging.info('7.Результат сформирован.')
        return preds[['ds', 'yhat_upper']].astype('str').to_json()
    except ValueError: 
        logging.info("Не удалось прочитать данные")
        return 'Ошибка чтения данных'
    except AttributeError:
        logging.info("Не удалось обработать ряд и сделать предсказание")
        return 'Ошибка обработки данных'





application = Flask(__name__)





@application.route('/TS/', methods=['POST'])

def get_data():

# Метод сервиса, получает тело запроса и возвращает прогноз верхней границы временного ряда.

    
    if not request.json:
        logging.info('Запрос не содержит данных')
        abort(400)
    data = request.json
    
    #print(data)
    try:
        return fb_predict(data, int(request.headers['n_period']), request.headers['freq'])
    except:
        abort(400)
        return 'error'

@application.route('/test/', methods=['GET'])

# Метод для тестирования отклика сервиса. Возвращает строку 'Hello' в случае доступности сервиса

def test_answer():    
    return 'Hello'


# In[ ]:


if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    #application.debug = True
    application.run(host='localhost', port='8081')       







