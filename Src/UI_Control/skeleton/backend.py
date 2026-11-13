"""Select a pose backend without importing the optional OpenMMLab stack."""


def create_estimator(model):
    if hasattr(model, 'create_estimator'):
        return model.create_estimator()
    from skeleton.detect_skeleton import PoseEstimater
    return PoseEstimater(model)
