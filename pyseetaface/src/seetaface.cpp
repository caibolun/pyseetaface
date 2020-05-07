/*
 * @Author: ArlenCai
 * @Date: 2020-03-11 20:41:05
 * @LastEditTime: 2020-05-07 16:52:50
 */
#pragma warning(disable: 4819)
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>

#include <seeta/Struct.h>
#include <seeta/CStruct.h>
#include <seeta/CFaceInfo.h>
#include <seeta/QualityAssessor.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace std;

typedef tuple<int, int, int, int> FaceRect;
typedef tuple<float, float> FacePoint;
typedef vector<FacePoint> FaceMark;
typedef vector<float> FaceFeature;

class SeetaFaceAPI
{
	protected:
		seeta::ModelSetting::Device device;
		int id;
		seeta::FaceDetector *FD;
		seeta::FaceLandmarker *FL81;
		seeta::FaceLandmarker *FL5;
		seeta::FaceRecognizer *FR;
		seeta::QualityAssessor QA;

	public:
		SeetaFaceAPI()
		{
			device = seeta::ModelSetting::CPU;
			id = 0;

		}
		~SeetaFaceAPI()
		{
			delete FD;
			delete FL81;
			delete FL5;
			delete FR;
		}

		void init(const char* fd_model, const char* fl81_model, const char* fl5_model, const char* fr_model, const char* _device, int _gpuid)
		{
			if(_device=="gpu") device = seeta::ModelSetting::GPU;
			else device = seeta::ModelSetting::CPU;
			id = _gpuid;
			detect_init(fd_model);
			align81_init(fl81_model);
			align5_init(fl5_model);
			extract_init(fr_model);
		}

		void detect_init(const char* fd_model)
		{
			seeta::ModelSetting FD_model(fd_model, device, id);
			FD = new seeta::FaceDetector(FD_model);
		}

		void align81_init(const char* fl_model)
		{
			seeta::ModelSetting FL_model(fl_model, device, id);
			FL81 = new seeta::FaceLandmarker(FL_model);
		}

		void align5_init(const char* fl_model)
		{
			seeta::ModelSetting FL_model(fl_model, device, id);
			FL5 = new seeta::FaceLandmarker(FL_model);
		}

		void extract_init(const char* fr_model)
		{
			seeta::ModelSetting FR_model(fr_model, device, id);
			FR = new seeta::FaceRecognizer(FR_model);
		}

		tuple<vector<FaceRect>, vector<float>> detect(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> img_array)
    	{
			SeetaImageData simage; 
			simage.height = int(img_array.shape(0));
			simage.width = int(img_array.shape(1));
			simage.channels = int(img_array.shape(2));
			simage.data = (unsigned char*) img_array.mutable_data();

			auto faces = FD->detect(simage);

			vector<FaceRect> face_rects;
			vector<float> face_scores;
			for (int i = 0; i < faces.size; ++i)
			{
				auto &face = faces.data[i];
				FaceRect face_rect = make_tuple(face.pos.x, face.pos.y, face.pos.width, face.pos.height);
            	face_rects.push_back(face_rect);
				face_scores.push_back(face.score);
			}
			return make_tuple(face_rects, face_scores);
    	}

		vector<FaceMark> align81(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> img_array, 
					vector<FaceRect> &face_rects)
    	{
			SeetaImageData simage; 
			simage.height = int(img_array.shape(0));
			simage.width = int(img_array.shape(1));
			simage.channels = int(img_array.shape(2));
			simage.data = (unsigned char*) img_array.mutable_data();

			vector<FaceMark> points81;
			for (int i = 0; i < face_rects.size(); i++)
        	{
				SeetaRect face_pos;
				face_pos.x = get<0>(face_rects[i]);
				face_pos.y = get<1>(face_rects[i]);
				face_pos.width = get<2>(face_rects[i]);
				face_pos.height = get<3>(face_rects[i]);

				auto points = FL81->mark(simage, face_pos);
				FaceMark p81;
				for (auto &point : points)
				{
                	p81.push_back(make_tuple(point.x, point.y));
				}
				points81.push_back(p81);
			}
			return points81;
    	}

		vector<FaceMark> align5(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> img_array, 
					vector<FaceRect> &face_rects)
    	{
			SeetaImageData simage; 
			simage.height = int(img_array.shape(0));
			simage.width = int(img_array.shape(1));
			simage.channels = int(img_array.shape(2));
			simage.data = (unsigned char*) img_array.mutable_data();

			vector<FaceMark> points5;
			for (int i = 0; i < face_rects.size(); i++)
        	{
				SeetaRect face_pos;
				face_pos.x = get<0>(face_rects[i]);
				face_pos.y = get<1>(face_rects[i]);
				face_pos.width = get<2>(face_rects[i]);
				face_pos.height = get<3>(face_rects[i]);

				auto points = FL5->mark(simage, face_pos);
				FaceMark p5;
				for (auto &point : points)
				{
                	p5.push_back(make_tuple(point.x, point.y));
				}
				points5.push_back(p5);
			}
			return points5;
    	}

	FaceFeature extract(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> img_array, FaceMark face_mark)
	{
		SeetaImageData simage; 
		simage.height = int(img_array.shape(0));
		simage.width = int(img_array.shape(1));
		simage.channels = int(img_array.shape(2));
		simage.data = (unsigned char*) img_array.mutable_data();

		SeetaPointF points[5];
		for(int i=0; i<face_mark.size(); i++)
		{
			points[i].x = get<0>(face_mark[i]);
			points[i].y = get<1>(face_mark[i]);
		}

		int feat_size = FR->GetExtractFeatureSize();
		float* p_feat = new float[feat_size];
		FR->Extract(simage, points, p_feat);
		
		FaceFeature feat;
		for(int i=0; i<feat_size; i++)
		{
			feat.push_back(p_feat[i]);
		}
		delete []p_feat;
		return feat;
	}

	tuple<int, float> evaluate(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> img_array,
			FaceRect face_rect, FaceMark face_mark, int face_size)
	{
		SeetaImageData simage; 
		simage.height = int(img_array.shape(0));
		simage.width = int(img_array.shape(1));
		simage.channels = int(img_array.shape(2));
		simage.data = (unsigned char*) img_array.mutable_data();

		SeetaRect face_pos;
		face_pos.x = get<0>(face_rect);
		face_pos.y = get<1>(face_rect);
		face_pos.width = get<2>(face_rect);
		face_pos.height = get<3>(face_rect);

		vector<SeetaPointF> points;
		for(int i=0; i<face_mark.size(); i++)
		{
			SeetaPointF p;
			p.x = get<0>(face_mark[i]);
			p.y = get<1>(face_mark[i]);
			points.push_back(p);
		}

		float score=0;
		QA.setFaceSize(face_size);
		int ret = QA.evaluate(simage, face_pos, points.data(), score);
		return make_tuple(ret, score);
	}
		
};

PYBIND11_MODULE(libseetaface, m) {
	pybind11::class_<SeetaFaceAPI>(m, "SeetaFaceAPI")
        .def(pybind11::init<>())
        .def("init", &SeetaFaceAPI::init)
		.def("detect_init", &SeetaFaceAPI::detect_init)
		.def("align81_init", &SeetaFaceAPI::align81_init)
		.def("align5_init", &SeetaFaceAPI::align5_init)
		.def("extract_init", &SeetaFaceAPI::extract_init)
        .def("detect", &SeetaFaceAPI::detect)
		.def("align81", &SeetaFaceAPI::align81)
		.def("align5", &SeetaFaceAPI::align5)
		.def("extract", &SeetaFaceAPI::extract)
		.def("evaluate", &SeetaFaceAPI::evaluate);
}