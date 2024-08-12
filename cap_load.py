import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pickle import load
import cv2

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess_img(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (1, vec.shape[1]))
    return vec


def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(image, beam_index):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = image
            preds = model.predict([e, np.array(par_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

# Load InceptionV3 model
base_model = InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)

# Load trained model and other necessary files
model = load_model('new-model-1.h5')
pickle_in = open("wordtoix.pkl", "rb")
wordtoix = load(pickle_in)
pickle_in = open("ixtoword.pkl", "rb")
ixtoword = load(pickle_in)
max_length = 74


def caption_this_image(image_path):
    image_encoding = encode(image_path)
    greedy_caption = greedy_search(image_encoding)
    beam_3_caption = beam_search(image_encoding,3)
    beam_5_caption = beam_search(image_encoding,5)

    # return {
    #     'image': image_path,
    #     'greedy_caption': greedy_caption,
    #     'beam_3_caption': beam_3_caption,
    #     'beam_5_caption': beam_5_caption
    # }

    # print( 'greedy_caption', greedy_caption)
    # print('beam_3_caption', beam_3_caption)
    # print('beam_5_caption', beam_5_caption)


def caption_this_image(image_path):
    # Load and display the input image
    image = cv2.imread(image_path)
    cv2.imshow('Input Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Encode the input image
    image_encoding = encode(image_path)

    # Generate captions using greedy search and beam search
    greedy_caption = greedy_search(image_encoding)
    beam_3_caption = beam_search(image_encoding, 3)
    beam_5_caption = beam_search(image_encoding, 5)

    # Print the generated captions
    print('Greedy Caption:',{ greedy_caption})
    print('Beam Search (Beam Size = 3) Caption:', {beam_3_caption})
    print('Beam Search (Beam Size = 5) Caption:', {beam_5_caption})

    # Display the generated captions
    cv2.putText(image, "Greedy Caption: " + greedy_caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "Beam Search (Beam Size = 3) Caption: " + beam_3_caption, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "Beam Search (Beam Size = 5) Caption: " + beam_5_caption, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output image with captions
    cv2.imshow('Output Image with Captions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path

image_path = './images/images.jpeg'
caption_this_image(image_path)

# data set flickres 8k data set
