import tensorflow as tf
import numpy as np
import PIL.Image


class NST:
    def __init__(self,
                 content_path: str = "../data/content/techweek.jpg",
                 style_path: str = "../data/style/starynightvangogh.jpg",
                 img_height: int = 1024,
                 epochs: int = 50):
        """

        Parameters
        ----------
        epochs : int 
        img_height : int
        style_path : str
        content_path : str

        """
        self.content_image = self.load_img(content_path, h=img_height)
        self.style_image = self.load_img(style_path, h=img_height)
        self.content_target = None
        self.style_target = None
        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet"
        )
        self.vgg.trainable = False
        self.generated_name = "{}_{}.jpg".format(
            content_path.replace("data", "Results").split(".jpg")[0],
            style_path.split("/")[-1].split(".")[0]
        )
        self.style_weight = 1e-2
        self.content_weight = 1e-1
        # noinspection PyUnresolvedReferences
        self.opt = tf.optimizers.Adam(
            learning_rate=0.01, beta_1=0.99, epsilon=1e-1)
        self.model = None
        self.epochs = epochs

    @staticmethod
    def tensor_to_image(tensor):
        """
        Coverts our tensor object to an image

        Parameters
        ----------
        tensor : tensorflow.python.ops.resource_variable_ops.ResourceVariable
        """
        # noinspection PyTypeChecker
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def save_image(self, img):
        """
        Save image to storage
        Parameters
        ----------
        img : tensorflow.python.ops.resource_variable_ops.ResourceVariable
        """
        img = self.tensor_to_image(img)
        img.save(self.generated_name)

    @staticmethod
    def load_img(path_to_img, h=512):
        """Loads an image from path and provides options normalise it

        Parameters
        ----------
        h : int
        path_to_img : str

        Returns
        -------
        img : tensorflow.python.framework.ops.EagerTensor
        """

        # noinspection PyUnresolvedReferences
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = h / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        # img = img[tf.newaxis, :]
        return img

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculate a Gram matrix that contains this information by calculating the feature vector's outer product with
        itself at each point and averaging that outer product across all locations.

        Parameters
        ----------
        input_tensor : KerasTensor

        Returns
        -------
        gram_matrix : KerasTensor
        """
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        g_matrix = tf.expand_dims(result, axis=0)
        input_shape = tf.shape(input_tensor)
        i_j = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return g_matrix / i_j

    def load_vgg(self):
        """
        Loads the VGG19 model to use it
        Returns
        -------
        None
        """
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_output = self.vgg.get_layer(content_layers[0]).output
        style_output = [self.vgg.get_layer(
            style_layer).output for style_layer in style_layers]
        gram_style_output = [self.gram_matrix(
            output_) for output_ in style_output]

        self.model = tf.keras.Model(
            [self.vgg.input], [content_output, gram_style_output])

    def loss_object(self, style_outputs, content_outputs):
        """

        Parameters
        ----------
        content_outputs : tensorflow.python.framework.ops.EagerTensor
        style_outputs : tensorflow.python.framework.ops.EagerTensor

        Returns
        -------
        total_loss : tensorflow.python.framework.ops.EagerTensor

        """
        content_loss = tf.reduce_mean(
            (content_outputs - self.content_target) ** 2)
        style_loss = tf.add_n([tf.reduce_mean((output_ - target_) ** 2)
                               for output_, target_ in zip(style_outputs, self.style_target)])
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

    def train_step(self, image, epoch):
        """
        Trains the Neural Style transfer model
        Parameters
        ----------
        epoch : int
        image : tensorflow.python.ops.resource_variable_ops.ResourceVariable

        """
        # noinspection PyUnresolvedReferences
        with tf.GradientTape() as tape:
            # noinspection PyTypeChecker
            output = self.model(image * 255)
            loss = self.loss_object(output[1], output[0])
        gradient = tape.gradient(loss, image)
        self.opt.apply_gradients([(gradient, image)])
        image.assign(tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=1.0))
        if epoch % 10 == 0:
            # noinspection PyUnresolvedReferences
            tf.print("Loss = {}".format(loss))

    def run(self):
        self.load_vgg()
        self.content_target = self.model(np.array([self.content_image * 255]))[0]
        self.style_target = self.model(np.array([self.style_image * 255]))[1]
        image = tf.image.convert_image_dtype(self.content_image, tf.float32)
        image = tf.Variable([image])
        for i in range(self.epochs):
            # noinspection PyTypeChecker
            self.train_step(image, i)
        # noinspection PyTypeChecker
        self.save_image(image)


if __name__ == '__main__':
    nst = NST(content_path="/content/data/content/techweek.jpg",
              style_path="/content/data/style/starynightvangogh.jpg",
              img_height=1024, epochs=10
    )
    nst.run()