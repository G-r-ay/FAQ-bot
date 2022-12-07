import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
import json

le = LabelEncoder()

with open('Intent.json') as chatfile:
    data = json.load(chatfile)

tags = []
inputs = []
responses = {}

for intention in data['intents']:
    responses[intention['intent']] = intention['responses']
    for lines in intention['text']:
        inputs.append(lines)
        tags.append(intention['intent'])

tag = le.fit_transform(tags)

LISA = tf.keras.models.load_model('LISA')

import random
while True:
  texts_p = []
  prediction_input = input('You : ')

  output = LISA.predict([prediction_input])
  output = output.argmax()
  #finding the right tag and predicting[
  response_tag = le.inverse_transform([output])[0]
  print("Going Merry : ",random.choice(responses[response_tag]))
  if output == "goodbye":
      break

