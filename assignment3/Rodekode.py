gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
img2 = cv2.drawKeypoints(gray, kp, img)
print(des[0])
# cv2.imshow('image', des)
# cv2.waitKey(2)
# cv2.destroyAllWindows()

(training_images, test_images) = splitFilesInDirectory('categories/accordion')
training_des_array = list(itertools.chain(*[descriptors(img) for img in training_images]))
kkm = train(200,training_des_array)
codebook = kkm.cluster_centers_
training_labels = kkm.predict(descriptors(training_images[0]))
test_labels = kkm.predict(descriptors(test_images[0]))
print(test_labels)


vectorizer = TfidfVectorizer()
table_list = [str(a[-1]) for a in table if a[2]==True]
vectorizer.fit(table_list)

cats = ['accordion', 'lobster', 'bass', 'panda', 'crocodile_head', 'brontosaurus', 'buddha', 'Faces']