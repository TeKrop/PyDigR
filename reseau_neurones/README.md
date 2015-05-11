# 5x7 Digit Image Recognition: Python (https://github.com/ayng/digit-recognition)

You need [Python Imaging Library](http://www.pythonware.com/products/pil/) to run this program.

## How to run

1. Run digit_maker.py in your terminal. This will generate noisy renditions of the clean PNGs in `training-sets/`.

        python digit_maker.py
        
2. Run digit_recog.py. To train on start, just type:

        python digit_recog.py
        
    After training is finished, input a save name. It will proceed to digit recognition.
    
3. If you want to load neuron data from a file (which is automatically saved in the `data/` directory after every training session):

        python digit_recog.py data/your_data_file
        
    Note that `data/your_data_file` could be changed to any location that leads to a [pickled](http://docs.python.org/2/library/pickle.html?highlight=pickle#pickle) Neuron list.
    
4. From there, `digit_recog.py` will let you know what to do.

## The theory

I'm using the following weight adjustment formulae:

    delta weight = learning_rate * error * input
    delta threshold = -(learning_rate * error)

And the activation function is a step function, 0 being not activated and 1 being activated. You can try changing the Neuron parameter called `a_func` in `digit_recog.py` to `"sigmoid"` to get a sigmoidal activation function, but I've tested this a few times and have gotten unpredictable results.

## The algorithm

1. There are 10 Neurons, each to be trained to its own digit (0-9).
2. There is 1 clean image of each number. Thirty noisy renditions of each number are generated. These 310 images constitute the training set.
3. Each neuron is fed 310 images (31 images per number for 10 numbers) and told whether to recognize it or not. For example, the neuron that is designated to recognize the number 1 will be told not to recognize 0, 2, 3, ... 9, and to recognize 1. It will loop back to the beginning of the training set until it stops making mistakes.
4. After training, the neuron should be fed an image that it has not seen before (simply generate a new noisy image) to test if the neuron has been properly trained.

---

There are a multitude of algorithms I could have used to hardcode number recognition at this low resolution, and probably in a simpler fashion as well. I chose to use a neural network so that I could familiarize myself with the theory and the algorithms before scaling it up to multiple layers and multiple neurons per layer for future problems that might have higher complexity. My Neuron class is written with this thought in mind.

Feel free to use any part of this repository for your own projects. Although the Neuron class is not comprehensive (as my artificial intelligence training is so far limited), I've put a lot of effort into making it versatile for the different usage scenarios that I am aware of.

---

Licensed under the MIT License. See `LICENSE` file for more details.
