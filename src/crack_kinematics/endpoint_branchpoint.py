import numpy as np
import mahotas as mh

def find_branch_points(skel):
    """Function to find branch points. This function is borrowed from
    https://stackoverflow.com/questions/43037692/how-to-find-branch-point-from-binary-skeletonize-image

    Args
    ----------
    skel: 2D numpy array, uint8
        The skeleton binary image as an array. The image range and dtype of the array must be [0, 1] and "uint8".

    Returns
    ----------
    bp : 2D numpy array
        An array of the same size as "skel" where the branch-point pixels have values of 1.
    """

    X = []
    # cross X
    X0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    X1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    # T like
    T = []
    # T0 contains X0
    T0 = np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    T1 = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])  # contains X1
    T2 = np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    T3 = np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    T4 = np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    T5 = np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    T6 = np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    T7 = np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    # Y like
    Y = []
    Y0 = np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])
    Y1 = np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]])
    Y2 = np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])
    Y3 = np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
    Y4 = np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        #bp = bp + mh.hit_or_miss(skel, x)
        bp = bp + mh.hitmiss(skel, x)
    #plt.figure()
    #plt.imshow(bp)
    for y in Y:
        #bp = bp + mh.hit_or_miss(skel, y)
        bp = bp + mh.hitmiss(skel, y)
    #plt.figure()
    #plt.imshow(bp)
    for t in T:
        #bp = bp + mh.hit_or_miss(skel, y)
        bp = bp + mh.hitmiss(skel, y)
    #plt.figure()
    #plt.imshow(bp)
    bp = np.where(bp == 0, bp, 255).astype('uint8')
    return bp


def find_end_points(skel):
    """Function to find end points of a skeleton binary image.
    This function is borrowed from the link below (slight modifications were applied):
    https://gist.github.com/jeanpat/5712699

    Args
    ----------
    skel : 2D numpy array, uint8
        The skeleton binary image as an array. The image range and dtype of the array must be [0, 1] and "uint8".

    Returns
    ----------
    end_points : 2D numpy array
        An array of the same size as "skel" where the end-point pixels have values of 1.
    """

    endpoint1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
    endpoint2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    endpoint3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])
    endpoint4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
    endpoint5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
    endpoint6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
    endpoint7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
    endpoint8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])

    ep1 = mh.morph.hitmiss(skel, endpoint1)
    ep2 = mh.morph.hitmiss(skel, endpoint2)
    ep3 = mh.morph.hitmiss(skel, endpoint3)
    ep4 = mh.morph.hitmiss(skel, endpoint4)
    ep5 = mh.morph.hitmiss(skel, endpoint5)
    ep6 = mh.morph.hitmiss(skel, endpoint6)
    ep7 = mh.morph.hitmiss(skel, endpoint7)
    ep8 = mh.morph.hitmiss(skel, endpoint8)
    end_points = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    end_points[end_points > 0] = 1
    return end_points
