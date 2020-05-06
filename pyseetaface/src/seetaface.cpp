/*
 * @Author: ArlenCai
 * @Date: 2020-03-11 20:41:05
 * @LastEditTime: 2020-03-16 10:50:14
 */
#pragma warning(disable: 4819)

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace std;

typedef tuple<int, int> Point;
typedef vector<Point> Mark;
typedef tuple<int, int, int, int> Rect;
typedef tuple<float, Rect, Mark> Face;

class SeetaFace()
{
	private:
		seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
		int id = 0;
		seeta::FaceDetector *FD;
		seeta::FaceLandmarker *PD;
		seeta::FaceRecognizer *FR;
	public:
		SeetaFace(const char* _device, int _gpuid)
		{
			if(_device=="gpu") 
		}
		~SeetaFace()
		{

		}
}

namespace py = pybind11;

PYBIND11_MODULE(seetaface, m) {
    m.def("detect", &detect);
    m.attr("__version__") = VERSION_INFO;
}