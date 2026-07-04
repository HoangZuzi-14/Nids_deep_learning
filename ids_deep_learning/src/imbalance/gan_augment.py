class GANTabularAugmenter:
    """Interface placeholder for CTGAN/WGAN-GP minority augmentation.

    The current notebook has a vanilla GAN prototype. Full project work should
    replace it with CTGAN, WGAN-GP, or conditional GAN and apply it only to
    train-set classes whose ratio is within the configured extreme-minority
    band.
    """

    def fit(self, X, y, target_classes):
        raise NotImplementedError("GAN augmentation interface is defined but not implemented.")

    def sample(self, n_samples: int, class_label):
        raise NotImplementedError("GAN augmentation interface is defined but not implemented.")

