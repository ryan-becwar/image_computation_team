import cv2, sys
import numpy as np


def stupidTrack(img, thresh):
    """given a grayscale image and threshold, find the center of the
    rectangle created to contain pixels below that threshold"""
    mask = np.argwhere(img < thresh)
    y1, x1 = mask[0]
    y2, x2 = mask[-1]
    return int((x1 + x2)/2), int((y1 + y2)/2)


def getGaussian(rows, cols, sigma):
    return cv2.getGaussianKernel(rows, sigma, cv2.CV_64F) @ \
           np.transpose(cv2.getGaussianKernel(cols, sigma, cv2.CV_64F))


def getGroundTruth(x, y, dims, gaussianDim, gaussianSigma):
    halfGaussDim = int(gaussianDim / 2)
    normal = np.zeros(dims)
    kernel = getGaussian(gaussianDim, gaussianDim, gaussianSigma) * 255
    normal[y - halfGaussDim:y + halfGaussDim, x - halfGaussDim:x + halfGaussDim] = kernel
    return normal


def getExactFilterCV(frame, groundTruth):
    lap = cv2.Laplacian(frame, cv2.CV_8U)
    dftF = cv2.dft(np.float32(lap), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftG = cv2.dft(np.float32(groundTruth), flags=cv2.DFT_COMPLEX_OUTPUT)
    #top = cv2.mulSpectrums(dftG, dftF, cv2.DFT_COMPLEX_OUTPUT, True)
    #bottom = cv2.mulSpectrums(dftF, dftF, cv2.DFT_COMPLEX_OUTPUT, True)
    #bottomReal = [bottom, np.zeros(bottom.shape)]
    #cv2.split(bottom, bottomReal)
    #bottomReal[1] = np.copy(bottomReal[0])
    #cv2.merge(bottomReal, bottom)
    dft = dftG / dftF
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])).astype('uint8')
    magnitude_spectrum = magnitude_spectrum / 255
    return magnitude_spectrum


def getExactFilterNP(frame, groundTruth):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_8U)
    #cv2.imshow('in', lap)
    dftF = np.fft.fft2(lap)
    #cv2.imshow('out', np.abs(np.fft.ifft2(dftF)).astype('uint8'))
    dftG = np.fft.fft2(groundTruth)
    dft = dftG / dftF
    #dft_shift = np.fft.fftshift(dft)
    #magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    #cv2.imshow('mag', magnitude_spectrum)
    #cv2.moveWindow('mag', 1100, 0)
    u = np.abs(np.fft.ifft2(dft)).astype('uint8')
    cv2.imshow('u', u)
    cv2.moveWindow('u', 1100, 600)
    return u


def getExactFilter(frame, groundTruth):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_8U)
    fourierF = np.fft.fft2(lap)
    fourierG = np.fft.fft2(groundTruth)
    fourierH = fourierG / fourierF
    spatialH = np.fft.ifft2(fourierH)
    shiftedH = np.fft.fftshift(np.abs(spatialH) * 3000)
    rotatedH = cv2.rotate(shiftedH, cv2.ROTATE_180)
    return rotatedH


def getAvgFilter(currAvg, newFilter):
    return newFilter


def getMaster(cropSize, frame, groundTruth, exactFilter, avgFilter):
    master = np.zeros((cropSize*2, cropSize*2, 3))
    master[0:cropSize, 0:cropSize,:] = frame / 255
    master[0:cropSize, cropSize:cropSize*2,0] = groundTruth
    master[0:cropSize, cropSize:cropSize*2,1] = groundTruth
    master[0:cropSize, cropSize:cropSize*2,2] = groundTruth
    master[cropSize:cropSize*2, 0:cropSize,0] = exactFilter
    master[cropSize:cropSize*2, 0:cropSize,1] = exactFilter
    master[cropSize:cropSize*2, 0:cropSize,2] = exactFilter
    master[cropSize:cropSize*2, cropSize:cropSize*2,0] = avgFilter
    master[cropSize:cropSize*2, cropSize:cropSize*2,1] = avgFilter
    master[cropSize:cropSize*2, cropSize:cropSize*2,2] = avgFilter
    return master


if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 3
    threshold = 32
    xOff = 350
    yOff = 100
    cropSize = 512
    avgFilter = None
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        cap.open()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[yOff:yOff+cropSize, xOff:xOff+cropSize]
        x, y = stupidTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), threshold)
        groundTruth = getGroundTruth(x, y, frame.shape[:-1], gaussianDim, gaussianSigma)
        exactFilter = getExactFilter(frame, groundTruth)
        avgFilter = getAvgFilter(avgFilter, exactFilter)
        master = getMaster(cropSize, frame, groundTruth, exactFilter, avgFilter)
        cv2.imshow('master', master)

        if cv2.waitKey(17) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


