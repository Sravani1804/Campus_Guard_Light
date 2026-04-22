import cv2

def compute_similarity(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)
