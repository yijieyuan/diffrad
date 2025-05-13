import numpy as np
import SimpleITK as sitk
from radiomics import glcm
from PIL import Image

def pyradiomics_glcm(image_array, low_bound, high_bound, bin_count, symmetricalGLCM, angle):
    image = sitk.GetImageFromArray(image_array)
   
    # Create mask (all ones)
    mask_array = np.ones(image_array.shape, dtype=np.int32)
    mask = sitk.GetImageFromArray(mask_array)
    settings = {
        'lowBound': low_bound,
        'highBound': high_bound,
        'binCount': bin_count,
        'symmetricalGLCM': symmetricalGLCM,
        'force2D': False,
        'distances': [max(angle)],
        # 'angles': [angle]  # Only one angle at a time
    }
    # Initialize GLCM calculator for this angle
    glcm_calculator = glcm.RadiomicsGLCM(image, mask, **settings)
    glcm_calculator.enableAllFeatures()
    glcm_calculator.execute()
    glcm_angles = glcm_calculator.angles
   
    for angle_idx, angle_values in enumerate(glcm_angles):
        if np.array_equal(angle_values, angle):
            break

    glcm_calculator.P_glcm = glcm_calculator.P_glcm[0, :, :, angle_idx][None, :, :, None]
    
    glcm_calculator._calculateCoefficients()
    features = {
        'Autocorrelation': glcm_calculator.getAutocorrelationFeatureValue(),
        'JointAverage': glcm_calculator.getJointAverageFeatureValue(),
        'ClusterProminence': glcm_calculator.getClusterProminenceFeatureValue(),
        'ClusterShade': glcm_calculator.getClusterShadeFeatureValue(),
        'ClusterTendency': glcm_calculator.getClusterTendencyFeatureValue(),
        'Contrast': glcm_calculator.getContrastFeatureValue(),
        'Correlation': glcm_calculator.getCorrelationFeatureValue(),
        'DifferenceAverage': glcm_calculator.getDifferenceAverageFeatureValue(),
        'DifferenceEntropy': glcm_calculator.getDifferenceEntropyFeatureValue(),
        'DifferenceVariance': glcm_calculator.getDifferenceVarianceFeatureValue(),
        'JointEnergy': glcm_calculator.getJointEnergyFeatureValue(),
        'JointEntropy': glcm_calculator.getJointEntropyFeatureValue(),
        'Imc1': glcm_calculator.getImc1FeatureValue(),
        'Imc2': glcm_calculator.getImc2FeatureValue(),
        'Idm': glcm_calculator.getIdmFeatureValue(),
        'MCC': glcm_calculator.getMCCFeatureValue(),
        'Idmn': glcm_calculator.getIdmnFeatureValue(),
        'Id': glcm_calculator.getIdFeatureValue(),
        'Idn': glcm_calculator.getIdnFeatureValue(),
        'InverseVariance': glcm_calculator.getInverseVarianceFeatureValue(),
        'MaximumProbability': glcm_calculator.getMaximumProbabilityFeatureValue(),
        'SumAverage': glcm_calculator.getSumAverageFeatureValue(),
        'SumEntropy': glcm_calculator.getSumEntropyFeatureValue(),
        'SumSquares': glcm_calculator.getSumSquaresFeatureValue()
    }
    return glcm_calculator.P_glcm, features
