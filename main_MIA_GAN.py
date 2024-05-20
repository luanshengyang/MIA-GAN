import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skfuzzy import mse

from MIA_GAN_Generator import Generator
from MIA_GAN_Discriminator import Discriminator
from data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
# from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers



# data path
path = "../../Data_Generation/Gan_Data/0.8/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"


# batch = 1 produces good results on U-NET
BATCH_SIZE = 32 # 32 64 128

# model
generator = Generator()
discriminator = Discriminator()
# optimizer
generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.9)
discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(2e-5)
#discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5)

"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L2 loss between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l2_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0

    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)

    return total_disc_loss


def huber_loss(target, gen_output, delta=2.0):
    error = target - gen_output
    abs_error = tf.abs(error)

    condition = tf.less(abs_error, delta)
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (abs_error - 0.5 * delta)

    loss = tf.where(condition, squared_loss, linear_loss)
    return tf.reduce_mean(loss)


def generator_loss(disc_generated_output, gen_output, target, huber_weight=110):
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)

    h_loss = huber_loss(target, gen_output, delta=2)
    total_gen_loss = tf.reduce_mean(gen_loss) + huber_weight * h_loss
    return total_gen_loss


def generated_image(model, test_input, tar, t=0):
    """Dispaly  the results of Generator"""
    prediction = model(test_input)
    #plt.figure(figsize=(15, 15))
    display_list = [np.squeeze(test_input[:,:,:,0]), np.squeeze(tar[:,:,:,0]), np.squeeze(prediction[:,:,:,0])]

    title = ['Input Y', 'Target H', 'Prediction H']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i]) 
        plt.axis("off")
    plt.savefig(os.path.join("generated_img", "batchsize_32_img_" + str(t) + ".png"))



def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target)   # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(epochs):
    nmse = []
    ep = []
    start_time = datetime.datetime.now()
    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)
        # train
        for bi, (target, input_image) in enumerate(load_image_train(path)):
            elapsed_time = datetime.datetime.now() - start_time
            gen_loss, disc_loss = train_step(input_image, target)
            print("B/E:", bi, '/' , epoch, ", Generator loss:", gen_loss.numpy(), ", Discriminator loss:", disc_loss.numpy(), ', time:',  elapsed_time )
        # generated and see the progress
        for bii, (tar, inp) in enumerate(load_image_test(path)):
            if bii == 100:
                generated_image(generator, inp, tar, t=epoch+1)

        ep.append(epoch + 1)
        realim, inpuim = load_image_test_y(path)   
        prediction = generator(inpuim)
        

        nmse.append(fuzz.nmse(np.squeeze(realim), np.squeeze(prediction)))
        print("NMSE:", nmse[epoch])


        if epoch == epochs - 1:
            nmse_epoch = TemporaryFile()
            np.save(nmse_epoch, nmse)

            # Save the predicted Channel
        matfiledata = {}  # make a dictionary to store the MAT data in
        matfiledata[u'predict_MIGan_batchsize_32_30epoch_40_dB_Indoor2p4_64ant_32users_8pilot'] = np.array(
            prediction)  # *** u prefix for variable name = unicode format, no issues thru Python 3.5; advise keeping u prefix indicator format based on feedback despite docs ***
        hdf5storage.write(matfiledata, '.',
                          'Results\Eest_MIGAN_' + str(epoch + 1) + '_batchsize_32_30epoch_40_dBIndoor2p4_64ant_32users_8pilot.mat',
                          matlab_compatible=True)

        plt.figure()
        plt.plot(ep, nmse, '^-r')
        plt.xlabel('Epoch')
        plt.ylabel('NMSE')
        figurepath = "shiyanjieguo/{}.png".format(str(epoch) + '_batchsize_32_30epoch_40_dB_64ant_32users_8pilot_NMSE')
        plt.savefig(figurepath)
        # plt.show();

    return nmse, mse, ep


if __name__ == "__main__":
    # train
    nmse, mse, ep = train(epochs=30)

    np.savetxt('NMSE_batchsize_32_MIA_Gan_40_dB_Indoor2p4_64ant_32subcarriers_8pilot loss.txt', nmse, fmt='%15f')
    np.savetxt('MSE_batchsize_32_MIA_Gan_40_dB_Indoor2p4_64ant_32subcarriers_8pilot loss.txt', mse, fmt='%15f')
    plt.xlabel('Epoch', fontproperties='Times New Roman')
    plt.xticks(range(0, 21, 5), fontproperties='Times New Roman')
    plt.ylabel('NMSE', fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.show()