import os.path
from tkinter.filedialog import askopenfilename

from keras import Sequential
from keras.applications import *
from keras.callbacks import *
from keras.layers import *
from keras.losses import *
from keras.models import load_model
from keras.optimizers import AdamW

from img_classify.dataset import *
from img_classify.downloader import *
from img_classify.evaluate_model import *
from img_classify.tools import *

# Tải dữ liệu từ thư viện Kaggle
KaggleDownloader(
	'tyre_quality',
	'warcoder/tyre-quality-classification'
)

# Thiết lập chung
train_path = 'datasets/tyre_quality/Digital images of defective and good condition tyres/'
img_save_path = 'imgs/tyre_quality/'
set_random_seed(69)
IMG_SIZE = (128, 128, 3)

# Tạo nhãn cho tập dữ liệu
class_names = CreateLabelFromDir(train_path).output

# Kiểm tra ảnh trong thư mục
CheckFiles(train_path)

# Kiểm tra độ cân bằng dữ liệu
CheckBalance(train_path, class_names, img_save_path=os.path.join(img_save_path, 'check_balance.jpg'))

# Tái cân bằng tập dữ liệu
img_model = ImageDataGenerator(
	horizontal_flip=True,
	vertical_flip=True,
	width_shift_range=.2,
	height_shift_range=.2,
	zoom_range=.2
)

fix_imbalance_with_image_augmentation(
	train_path,
	img_model=img_model,
	img_size=(IMG_SIZE[0], IMG_SIZE[1]),
	class_names=class_names
)

def training_model():
	## Nạp ảnh
	a = ImagestoArray(train_path, class_names, (IMG_SIZE[0], IMG_SIZE[1]))
	images, labels = a.images, a.labels

	## Tách mảng để Train/Test/Val (65/20/15)
	(train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = TrainTestValSplit(
		images,
		labels,
		train_size=.70,
		test_size=.25,
		val_size=.05
	).output

	## Tăng cường ảnh tập Train
	train_img_model = ImageDataGenerator(
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=.2,
		rotation_range=30,
		shear_range=.2
	)
	train_img_model.flow(
		train_images,
		train_labels,
		seed=69
	)

	## Rescale ảnh
	train_images = train_images / 255.
	test_images = test_images / 255.
	val_images = val_images / 255.

	# Xây dựng mô hình
	def my_model():
		#  Thiết lập mô hình
		base_model = MobileNetV2(include_top=False, input_shape=IMG_SIZE)
		base_model.trainable = False
		model = Sequential(
			[
				base_model,

				Flatten(),
				Dense(128, activation='relu', kernel_regularizer='l2'),
				Dropout(.2),
				Dense(64, activation='relu'),
				Dropout(.2),
				Dense(len(class_names), activation='softmax')
			]
		)
		# Biên dịch mô hình
		model.compile(
			AdamW(),
			CategoricalCrossentropy(),
			['accuracy']
		)
		return model

	# Chọn mô hình
	model = my_model()
	# Khái quát mô hình
	model.summary()
	# Đào tạo mô hình
	history = model.fit(
		train_images,
		train_labels,
		epochs=100,
		validation_data=(test_images, test_labels),
		callbacks=[
			EarlyStopping(
				monitor='val_accuracy',
				patience=10,
				restore_best_weights=True
			),
			ReduceLROnPlateau(
				monitor='val_accuracy',
				factor=.1,
				patience=5,
				min_lr=.0001
			)
		]
	)
	# Lưu mô hình
	model.save('models/tyre_quality_model.h5')
	# Đánh giá quá trình đào tạo
	EvalofTraining(history, img_save_path)
	# Đánh giá mô hình qua tập Test
	sai_so, do_chinh_xac = model.evaluate(test_images, test_labels)
	print(f'Sai số: {sai_so}')
	print(f'Độ chính xác: {do_chinh_xac}')
	# Dự đoán 4 ảnh trong tập Val
	pred_labels = model.predict(val_images)
	eval_of_model_with_images(
		3, 3, val_images, pred_labels, val_labels, class_names,
		img_save_path=os.path.join(img_save_path, 'eval_of_model_with_images.jpg')
	)
	heatmap_plot(
		val_labels, pred_labels, class_names,
		img_save_path=os.path.join(img_save_path, 'eval_of_model_with_heatmap.jpg')
	)

## => Result: Accuracy ~ 91%

def predict():
	model = load_model('models/tyre_quality_model.h5')
	file = askopenfilename()
	with open(file) as f:
		image = f.convert('RGB')
		image = image.resize((128, 128))
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
	pred = model.predict(image)
	print(class_names[np.argmax(pred)])

training_model()
# predict()