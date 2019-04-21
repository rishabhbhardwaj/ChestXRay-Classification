from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        iaa.Noop(),
    ],
    random_order=True,
)
