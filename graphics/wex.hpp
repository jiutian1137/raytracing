#pragma once

/// Windows Extension 
///@license Free 
///@review 2022-5-29 
///@contact Jiang1998Nan@outlook.com 
#define _WINDOWS_EXTENSION_

#include <windows.h>
#undef min
#undef max
#undef near
#undef far
#include <chrono>
#include <cassert>
//#include <iostream>

/// multi-thread.
#include <thread>
#include <mutex>

#ifdef _OPENGL_LIBRARY_HIGHLEVEL_
#include "opengl/library2.hpp"
#endif

///@diagram
/// +-----------------------+
/// |  message_queue::run() |
/// +-----------+-----------+
///             |-- -- -- -- -- -- -- -- -- -- -+ -- -- -- -- -- -- -- -- -- -+
///             |                               |                             |
/// +-----------|-------------+    +------------|----------+    +-------------|-----------+
/// |  message_queue::start() | -> |       Message-Loop    | -> |  message_queue::stop()  |
/// +-------------------------+    +---------+----\--------+    +-------------------------+
///    Multithreads may be                 /        \              Therefore, multithreads must
///      start in here                    |         |                stop before the destructor, is here.
/// 	                                     \       / This is a loop.
///                                           -- 
///                           +----------------------------------+ hwnd +------------------------------------+
/// 	                        | message_queue::process:message() |  ->  | window_userdata::process_message() |
///                           +----------------------------------+ hwnd +------------------------------------+
///                           | message_queue::wait_message()    |  ->  | window_userdata::wait_message()    |
///                           +----------------------------------+      +------------------------------------+
namespace wex {
	using message = MSG;
	class message_queue;

	class window {
	public:
		::HWND operator&() const {
			return (::HWND)this;
		}

		bool belong_class(LPCWSTR that_cname) {
			wchar_t this_cname[256];
			int length = GetClassName((HWND)this, this_cname, 256); assert(length != 0);
			return lstrcmp(this_cname, that_cname) == 0;
		}

		std::pair<LONG,LONG> get_windowsize() const {
			RECT rect;
			GetWindowRect((HWND)this, &rect);
			return { rect.right - rect.left, rect.bottom - rect.top };
		}

		std::pair<LONG,LONG> get_clientsize() const {
			RECT rect;
			GetClientRect((HWND)this, &rect);
			return { rect.right - rect.left, rect.bottom - rect.top };
		}

		std::wstring get_name() const {
			auto name = std::wstring(GetWindowTextLength((HWND)this)+1, L'\0');
			name.resize( GetWindowText((HWND)this, name.data(), (int)name.size()) );
			return name;
		}
	
		bool set_name(LPCWSTR name) {
			return SetWindowText((HWND)this, name);
		}
	
		bool show(int mCmdShow = SW_SHOW) {
			return ShowWindow((HWND)this, mCmdShow);
		}
	
		bool flash(bool bInvert = false) {
			return FlashWindow((HWND)this, bInvert);
		}

		bool update() {
			return UpdateWindow((HWND)this);
		}

		struct userdata {
			::HWND handler;
			userdata() : handler(nullptr) {};
			explicit userdata(wex::window& the_window) : handler((HWND)(&the_window)) {}
			virtual ~userdata() noexcept {}
			virtual void start(wex::message_queue& sender) {}
			virtual void process_message(const wex::message& msg, wex::message_queue& sender) {}
			virtual void wait_message(wex::message_queue& sender) {}
			virtual void stop() {}
			wex::window& window() const { return *((wex::window*)handler); }

			template<typename concret_userdata> requires std::is_base_of_v<userdata, concret_userdata>
			concret_userdata& as() { return dynamic_cast<concret_userdata&>( *this ); }

			template<typename concret_userdata> requires std::is_base_of_v<userdata, concret_userdata>
			operator concret_userdata&() { return dynamic_cast<concret_userdata&>( *this ); }
		};

		wex::window::userdata* get_userdata() {
			return (wex::window::userdata*)GetWindowLongPtr((HWND)this, GWLP_USERDATA);
		}

		void set_userdata(wex::window::userdata* the_user_data) {
			auto prev_user_data = (wex::window::userdata*)SetWindowLongPtr((HWND)this, GWLP_USERDATA, (LONG_PTR)the_user_data);
			if (prev_user_data) {
				delete prev_user_data;
			}
		}
	};

	class message_queue {
	public:
		::DWORD thread_id = 0;
		::HINSTANCE app_handler = nullptr;
		std::wstring default_class_name;
		std::ostream* logger = nullptr;
		std::chrono::steady_clock::time_point start_time;
		std::chrono::steady_clock::time_point current_time;
		std::chrono::steady_clock::time_point prev_time;
	
		/// Process all messages(KEY,MOUSE,MOVE,SIZE,...,USER), this is static,
		///@note want to no-static process message, @see *::run(), *::process_message(..).
		static LRESULT (CALLBACK static_process_message)(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
			switch (msg) { 

			case WM_CREATE:
				//std::cout << "WM_CREATE," << hwnd << std::endl;
				return DefWindowProc(hwnd, msg, wParam, lParam);

			case WM_CLOSE:
				/// many kinds of close. we post this message out.
				/// "https://docs.microsoft.com/en-us/windows/win32/learnwin32/closing-the-window"
				PostMessage(hwnd, WM_USER|WM_CLOSE, wParam, lParam);
				return 0;

			case WM_DESTROY:
				if (((wex::window*)hwnd)->get_userdata()) {
					((wex::window*)hwnd)->get_userdata()->stop();
					((wex::window*)hwnd)->set_userdata(nullptr);
				}
				return DefWindowProc(hwnd, msg, wParam, lParam);

			case WM_EXITSIZEMOVE:
				///
				/// WM_SIZE doesnt work as expected
				///
				/// Answer
				/// PeekMessage() can only see messages that were posted to the message queue. That excludes WM_SIZE,
				/// it is sent, not posted. Or in other words, it is delivered by SendMessage(), it calls the window
				/// procedure directly and bypasses the message queue. So yes, your code starts working because you
				/// now repost the message with PostMessage, it is put on the message queue so PeekMessage can see it.
				///
				/// Something different happens when the user resizes the window. That's reported by another message:
				/// WM_SIZING. It is generated, at a pretty high rate, when Windows starts a modal message loop to
				/// implement the resizing operation. It gives due notice of this, you'll get the WM_ENTERSIZEMOVE
				/// when the modal loop starts (user clicks a window corner), WM_EXITSIZEMOVE when it is complete
				/// (user releases the button). You'll get a bunch of WM_SIZING messages, sent to your window procedure.
				/// Not posted. And one WM_SIZE to give the final size. One way to not see these reflected versions of
				/// these messages is when you call PeekMessage() in your own message loop. It won't be called when
				/// the Windows modal resize loop is active.
				///
				/// Hard to give better advice, it is really unclear why you are doing this. The "doctor, it hurts,
				/// don't do it then" medical answer is highly likely to be relevant. I suspect you might want to
				/// reflect the WM_SIZING message as well. The largest issue is that by the time you retrieve these
				/// messages from the queue, the window size has already changed and the notification is just plain
				/// stale. Which is why the message is sent and not posted.
				///                                                                    answered Jun 11 '12 at 17:31
				///                                                                    Hans Passant
				///                                                  "https://stackoverflow.com/a/10984802/16974723"
				PostMessage(hwnd, WM_USER|WM_EXITSIZEMOVE, wParam, lParam);
				return DefWindowProc(hwnd, msg, wParam, lParam);

			default:
				return DefWindowProc(hwnd, msg, wParam, lParam);

			}
		}

		/// Message-Loop.
		virtual int run() {
			this->start_time = std::chrono::steady_clock::now();
			this->prev_time = this->start_time;
			this->start();

			wex::message msg = {0};
			while(msg.message != WM_QUIT) {
				this->current_time = std::chrono::steady_clock::now();
				if ( PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE) ) {
					TranslateMessage(&msg);
					DispatchMessage(&msg);
					this->process_message(msg);
				} else {
					this->wait_message();
				}
				this->prev_time = this->current_time;
			}

			this->stop();
			return 0;
		}

		#define _For_each_userdata(in_lParam, userdata_name, do_something) \
		EnumThreadWindows(this->thread_id, \
			[](HWND hwnd, LPARAM lParam) { \
				auto userdata_name = ((wex::window*)hwnd)->get_userdata(); \
				if (userdata_name) { do_something; } \
				return TRUE; \
			}, \
			in_lParam \
		)

		typedef void (CALLBACK* DEFAULT_USER_MESSAGE_WPARAM)(wex::message_queue&, LPARAM);
	
		virtual void start() { 
			_For_each_userdata((LPARAM)this, wudi, wudi->start(*((wex::message_queue*)lParam)));
		}

		virtual void process_message(const MSG& msg) {
			/*if (msg.message == WM_USER && msg.wParam != 0) {
				assert(msg.hwnd == nullptr);
				((DEFAULT_USER_MESSAGE_WPARAM)msg.wParam)(*this, msg.lParam);
				return;
			}*/

			if (msg.hwnd) {
				auto wudi = (wex::window::userdata*)GetWindowLongPtr(msg.hwnd, GWLP_USERDATA);
				if (wudi) { 
					wudi->process_message(msg, *this); 
				}
			} else {
				auto lParamimpl = std::pair<const wex::message&, wex::message_queue&>(msg, *this);
				_For_each_userdata((LPARAM)(&lParamimpl), wudi,
					auto lParamimplX = (std::pair<const wex::message&, wex::message_queue&> *)lParam;
					wudi->process_message(lParamimplX->first, lParamimplX->second)
				);
			}
		}

		virtual void wait_message() {
			_For_each_userdata((LPARAM)this, wudi, wudi->wait_message(*((wex::message_queue*)lParam)));
		}
	
		virtual void stop() {
			_For_each_userdata((LPARAM)0, wudi, wudi->stop());
		}

		#undef _For_each_userdata

		message_queue() = default;

		message_queue(HINSTANCE hInstance, LPCWSTR lpszClassName = L"default", std::ostream* logger = nullptr)
			: thread_id(GetCurrentThreadId()), app_handler(hInstance), default_class_name(lpszClassName), logger(logger) {
			WNDCLASS wc;
			wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
			wc.lpfnWndProc   = &(wex::message_queue::static_process_message);
			wc.cbClsExtra    = 0;
			wc.cbWndExtra    = 0;
			wc.hInstance     = hInstance;
			wc.hIcon         = LoadIcon(0, IDI_APPLICATION);
			wc.hCursor       = LoadCursor(0, IDC_CROSS);
			wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
			wc.lpszMenuName  = nullptr;
			wc.lpszClassName = lpszClassName;
			auto wc_ok = RegisterClass(&wc);
			assert( wc_ok );
		}

		virtual ~message_queue() noexcept {
			auto lParam = (LPARAM)default_class_name.c_str();
			auto lpfn = [](HWND hwnd, LPARAM lParam) {
				if (((wex::window*)hwnd)->belong_class((wchar_t*)lParam)) {
					((wex::window*)hwnd)->set_userdata(nullptr);
					DestroyWindow(hwnd);
				}
				return TRUE;
			};
			EnumWindows(lpfn, lParam);
			UnregisterClass(this->default_class_name.c_str(), this->app_handler);
		}

	public:
		wex::window* find_window(LPCWSTR name) {
			return (wex::window*)FindWindow(this->default_class_name.c_str(), name);
		}

		wex::window* create_window(LPCWSTR name, DWORD style, int x, int y, int width, int height) {
			if (GetCurrentThreadId() == this->thread_id) {
				RECT client_rect = { 0, 0, width, height };
				AdjustWindowRect(&client_rect, style, false);
				return (wex::window*)CreateWindow(
					this->default_class_name.c_str(), 
					name,
					style, 
					x, y, 
					client_rect.right - client_rect.left, client_rect.bottom - client_rect.top,
					nullptr,
					nullptr, 
					this->app_handler,
					0
				);
			} else {
				auto res        = std::atomic<wex::window*>(nullptr);
				auto lParamimpl = std::tuple<std::atomic<wex::window*>&, LPCWSTR, DWORD, int, int, int, int>(res, name, style, x, y, width, height);
				DEFAULT_USER_MESSAGE_WPARAM f = [](wex::message_queue& q, LPARAM lParam) {
					auto* lParamimpl = (std::tuple<std::atomic<wex::window*>&, LPCWSTR, DWORD, int, int, int, int> *)lParam;
					std::get<0>(*lParamimpl).store(
						q.create_window(std::get<1>(*lParamimpl), std::get<2>(*lParamimpl),
							std::get<3>(*lParamimpl), std::get<4>(*lParamimpl), std::get<5>(*lParamimpl), std::get<6>(*lParamimpl)) );
					std::get<0>(*lParamimpl).notify_one();
				};
				PostThreadMessage(this->thread_id, WM_USER, (WPARAM)(f), (LPARAM)(&lParamimpl));
				res.wait(nullptr);
				return res.load();
			}
		}

		template<typename concret_userdata, typename... types> requires std::is_base_of_v<wex::window::userdata, concret_userdata>
		wex::window* create_window_and_userdata(LPCWSTR name, DWORD style, int x, int y, int width, int height, types&&... args) {
			auto the_window = this->create_window(name, style, x, y, width, height);
			the_window->set_userdata(new concret_userdata(*the_window, std::forward<types>(args)...));
			return the_window;
		}
	
		wex::window::userdata* operator[](LPCWSTR name) {
			auto the_window = (wex::window*)FindWindow(this->default_class_name.c_str(), name);
			assert( the_window );
			return the_window->get_userdata();
		}
	};

	template<typename window_userdata>
	class basic_event : public window_userdata {
		using base = window_userdata;
	public:
		::POINT prev_mouse_pos = { -1, -1 };

		using base::base;

		virtual void start(wex::message_queue& sender) override {
			base::start(sender);
			this->prev_mouse_pos = { -1, -1 };
		}

		virtual void process_message(const wex::message& msg, wex::message_queue& sender) override {
			base::process_message(msg, sender);
			//if (sender.logger) {
			//	(*sender.logger) << std::format("message:{{ hwnd:{0}, msg:{1}, wParam:{2}, lParam:{3} }}",
			//		(void*)msg.hwnd, msg.message, (UINT)msg.wParam, (UINT)msg.lParam) << std::endl;
			//}

			/// Why are messages made into interfaces?
			/// 
			///		process_mouse_message() process_keyboard_message() ...
			/// 
			/// 1. Reuse will lead to many redundant process.
			///	
			///		/// Base::process_message
			///		switch(msg->message){ case WM_KEYDOWN:... case WM_MOUSEMOVE:... ... }
			///		/// process_message
			///		switch(msg->message){ case WM_KEYDOWN:... case WM_MOUSEMOVE:... ... }
			/// 
			/// 2. indirect custom param cannot placed in actual application(..) and members of class(..).
			/// 
			///		MOUSEMOVE_DELTA ...
			/// 
			if (WM_MOUSEFIRST <= msg.message && msg.message <= WM_MOUSELAST) {
				this->process_mouse_message(msg, sender);
			} else if (WM_KEYFIRST <= msg.message && msg.message <= WM_KEYLAST) {
				this->process_keyboard_message(msg, sender);
			} else if (msg.message == (WM_USER|WM_CLOSE)) {
				this->process_close_message(msg, sender);
			} else if (msg.message == (WM_USER|WM_EXITSIZEMOVE)) {
				this->process_exit_sizemove_message(msg, sender);
			} else {
				this->process_other_message(msg, sender);
			}
		}

		#define GET_MOUSE_POINT_LPARAM(lParam) POINT{ (LONG)((short)LOWORD(lParam)), (LONG)((short)HIWORD(lParam)) }

		virtual void process_mouse_message(const wex::message& msg, wex::message_queue& sender) {
			if (msg.message == WM_MOUSEMOVE) {
				if (this->prev_mouse_pos.x == -1) {
					this->prev_mouse_pos = GET_MOUSE_POINT_LPARAM(msg.lParam);
					this->process_mouse_move_message(msg, 0, 0, sender);
				} else {
					POINT current_mouse_pos = GET_MOUSE_POINT_LPARAM(msg.lParam);
					this->process_mouse_move_message(msg, current_mouse_pos.x - this->prev_mouse_pos.x, current_mouse_pos.y - this->prev_mouse_pos.y, sender);
					this->prev_mouse_pos = current_mouse_pos;
				}
			} else if (msg.message == WM_MOUSEWHEEL) {
				this->process_mouse_wheel_message(msg, sender);
			}
		}
	
		virtual void process_mouse_move_message(const wex::message& msg, LONG dx, LONG dy, wex::message_queue& sender) {}

		virtual void process_mouse_wheel_message(const wex::message& msg, wex::message_queue& sender) {}

		virtual void process_keyboard_message(const wex::message& msg, wex::message_queue& sender) {}
	
		virtual void process_close_message(const wex::message& msg, wex::message_queue& sender) { DestroyWindow(base::handler); }

		virtual void process_exit_sizemove_message(const wex::message& msg, wex::message_queue& sender) {}

		virtual void process_other_message(const wex::message& msg, wex::message_queue& sender) {}
	};

	template<typename window_userdata>
	class mainwindow : public wex::basic_event<window_userdata> {
		using base = wex::basic_event<window_userdata>;
	public:

		using base::base;

		virtual void process_close_message(const wex::message& msg, wex::message_queue&) override {
			if (MessageBox(msg.hwnd, L"Really Quit?", L"Close", MB_OKCANCEL) == IDOK) {
				PostQuitMessage(0);
			}
		}
	};

	template<typename window_userdata>
	class multithread_rendering : public window_userdata {
		using base = window_userdata;
	public:
		static constexpr size_t RENDERING = 0, OTHER = 1, THREAD_COUNT = 2;
		std::thread threads[THREAD_COUNT];
		std::atomic_flag running;
		std::atomic_flag waiting;
		std::atomic_uint next_frames;

		static constexpr size_t /*RENDERING = 0, */UPLOAD = 1, MUTEX_COUNT = 2;
		std::mutex mutexs[MUTEX_COUNT];

		using base::base;

		virtual void render(wex::message_queue& q) {}

		virtual void start(wex::message_queue& q) override {
			base::start(q);

			running.test_and_set();
			waiting.clear();
			next_frames.store(1);
			threads[RENDERING] = std::thread([&]() {
				///		running -> next_frames -> waiting -> running ...  (0)
				/// Error, because 'next_frames' may set 'waiting' again and no process can clear 'waiting'.
				/// So we should clear 'next_frames' first, but the process of 'next_frames' influence 'waiting' is not atomic unless more complexity.
				/// 
				///		running -> waiting -> next_frames -> running ...  (1)
				///		waiting -> running -> next_frames -> waiting ...  (2) error.
				///		waiting -> next_frames -> running -> waiting ...  (3) same as (1).
				///		next_frames -> running -> waiting -> next_frames . (4) same as (1).
				///		next_frames -> waiting -> running -> next_frames . (4) error same as (2).
				/// In fact, we need to worry about only "waiting", because it may reset automatically.
				/// So we just have to break the relation.
				while (running.test()) {
					waiting.wait(true);
					--next_frames;

					render(q);

					if (next_frames == 0) {
						waiting.test_and_set();
					}
				}
			});
		}

		virtual void stop() override {
			running.clear();
			next_frames.store(9999);
			waiting.clear();
			waiting.notify_one();
			for (size_t i = 0; i != THREAD_COUNT; ++i) {
				if (threads[i].joinable()) {
					threads[i].join();
				}
			}

			base::stop();
		}
	};

	class window_opengl : public wex::window::userdata {
	public:
		::HDC context = nullptr;

		static PIXELFORMATDESCRIPTOR default_pixelformat_descriptor(bool depth = true) {
			return PIXELFORMATDESCRIPTOR{
				sizeof(PIXELFORMATDESCRIPTOR),
				1,
				PFD_DRAW_TO_WINDOW|PFD_DOUBLEBUFFER|PFD_SUPPORT_OPENGL,
				PFD_TYPE_RGBA,
				32,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				BYTE(depth?32:0), 0,
				0, 
				PFD_MAIN_PLANE, 
				0, 
				0, 0, 0
			};
		}
	
		window_opengl() = default;
	
		explicit 
		window_opengl(wex::window& the_window, PIXELFORMATDESCRIPTOR pfd = default_pixelformat_descriptor()) : wex::window::userdata(the_window),
			context(::GetDC(wex::window::userdata::handler)) {
			auto pfd_ok = SetPixelFormat(this->context, ChoosePixelFormat(this->context, &pfd), &pfd);
			assert( pfd_ok );

			//BOOL(WINAPI* wglChoosePixelFormatARB) (HDC hdc, const int* piAttribIList, const FLOAT* pfAttribFList, UINT nMaxFormats, int* piFormats, UINT* nNumFormats);
			///*BOOL(WINAPI* wglGetPixelFormatAttribfvARB) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, FLOAT* pfValues);
			//BOOL(WINAPI* wglGetPixelFormatAttribivARB) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, int* piValues);*/
			//wglChoosePixelFormatARB = (decltype(wglChoosePixelFormatARB))wglGetProcAddress("wglChoosePixelFormatARB");
			//if (wglChoosePixelFormatARB != nullptr && samplers != 1) {
			//	constexpr int WGL_NUMBER_PIXEL_FORMATS_ARB = 0x2000;
			//	constexpr int WGL_DRAW_TO_WINDOW_ARB = 0x2001;
			//	constexpr int WGL_DRAW_TO_BITMAP_ARB = 0x2002;
			//	constexpr int WGL_ACCELERATION_ARB = 0x2003;
			//	constexpr int WGL_NEED_PALETTE_ARB = 0x2004;
			//	constexpr int WGL_NEED_SYSTEM_PALETTE_ARB = 0x2005;
			//	constexpr int WGL_SWAP_LAYER_BUFFERS_ARB = 0x2006;
			//	constexpr int WGL_SWAP_METHOD_ARB = 0x2007;
			//	constexpr int WGL_NUMBER_OVERLAYS_ARB = 0x2008;
			//	constexpr int WGL_NUMBER_UNDERLAYS_ARB = 0x2009;
			//	constexpr int WGL_TRANSPARENT_ARB = 0x200A;
			//	constexpr int WGL_SHARE_DEPTH_ARB = 0x200C;
			//	constexpr int WGL_SHARE_STENCIL_ARB = 0x200D;
			//	constexpr int WGL_SHARE_ACCUM_ARB = 0x200E;
			//	constexpr int WGL_SUPPORT_GDI_ARB = 0x200F;
			//	constexpr int WGL_SUPPORT_OPENGL_ARB = 0x2010;
			//	constexpr int WGL_DOUBLE_BUFFER_ARB = 0x2011;
			//	constexpr int WGL_STEREO_ARB = 0x2012;
			//	constexpr int WGL_PIXEL_TYPE_ARB = 0x2013;
			//	constexpr int WGL_COLOR_BITS_ARB = 0x2014;
			//	constexpr int WGL_RED_BITS_ARB = 0x2015;
			//	constexpr int WGL_RED_SHIFT_ARB = 0x2016;
			//	constexpr int WGL_GREEN_BITS_ARB = 0x2017;
			//	constexpr int WGL_GREEN_SHIFT_ARB = 0x2018;
			//	constexpr int WGL_BLUE_BITS_ARB = 0x2019;
			//	constexpr int WGL_BLUE_SHIFT_ARB = 0x201A;
			//	constexpr int WGL_ALPHA_BITS_ARB = 0x201B;
			//	constexpr int WGL_ALPHA_SHIFT_ARB = 0x201C;
			//	constexpr int WGL_ACCUM_BITS_ARB = 0x201D;
			//	constexpr int WGL_ACCUM_RED_BITS_ARB = 0x201E;
			//	constexpr int WGL_ACCUM_GREEN_BITS_ARB = 0x201F;
			//	constexpr int WGL_ACCUM_BLUE_BITS_ARB = 0x2020;
			//	constexpr int WGL_ACCUM_ALPHA_BITS_ARB = 0x2021;
			//	constexpr int WGL_DEPTH_BITS_ARB = 0x2022;
			//	constexpr int WGL_STENCIL_BITS_ARB = 0x2023;
			//	constexpr int WGL_AUX_BUFFERS_ARB = 0x2024;
			//	constexpr int WGL_NO_ACCELERATION_ARB = 0x2025;
			//	constexpr int WGL_GENERIC_ACCELERATION_ARB = 0x2026;
			//	constexpr int WGL_FULL_ACCELERATION_ARB = 0x2027;
			//	constexpr int WGL_SWAP_EXCHANGE_ARB = 0x2028;
			//	constexpr int WGL_SWAP_COPY_ARB = 0x2029;
			//	constexpr int WGL_SWAP_UNDEFINED_ARB = 0x202A;
			//	constexpr int WGL_TYPE_RGBA_ARB = 0x202B;
			//	constexpr int WGL_TYPE_COLORINDEX_ARB = 0x202C;
			//	constexpr int WGL_TRANSPARENT_RED_VALUE_ARB = 0x2037;
			//	constexpr int WGL_TRANSPARENT_GREEN_VALUE_ARB = 0x2038;
			//	constexpr int WGL_TRANSPARENT_BLUE_VALUE_ARB = 0x2039;
			//	constexpr int WGL_TRANSPARENT_ALPHA_VALUE_ARB = 0x203A;
			//	constexpr int WGL_TRANSPARENT_INDEX_VALUE_ARB = 0x203B;
			//	constexpr int  WGL_SAMPLE_BUFFERS_ARB = 0x2041;
			//	constexpr int  WGL_SAMPLES_ARB = 0x2042;

			//	int pixelFormat; UINT numFormats;
			//	int piAttribIList[] = { 
			//		WGL_DRAW_TO_WINDOW_ARB,TRUE,
			//		WGL_SUPPORT_OPENGL_ARB,TRUE,
			//		WGL_DOUBLE_BUFFER_ARB,TRUE,
			//		WGL_COLOR_BITS_ARB,32, WGL_RED_BITS_ARB,8, WGL_GREEN_BITS_ARB,8, WGL_BLUE_BITS_ARB,8, WGL_ALPHA_BITS_ARB,8,
			//		WGL_DEPTH_BITS_ARB,24, WGL_STENCIL_BITS_ARB,8,
			//		WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
			//		WGL_SAMPLE_BUFFERS_ARB,TRUE, WGL_SAMPLES_ARB,4,
			//		0, 0 
			//	};
			//	float pfAttribFList[] = { 
			//		0, 0 
			//	};

			//	RECT R = { 0, 0, width, height }; AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
			//	HWND hMainWindowARB = CreateWindow(
			//		lpszClassName,
			//		lpMainWindowName,
			//		WS_OVERLAPPEDWINDOW, 
			//		CW_USEDEFAULT, 
			//		CW_USEDEFAULT, 
			//		R.right - R.left,
			//		R.bottom - R.top,
			//		0,
			//		0,
			//		hInstance, 
			//		0
			//	);
			//	if (hMainWindowARB == nullptr) {
			//		throw std::exception("hMainWindow == nullptr");
			//	}
			//	HDC hDeviceARB = GetDC(hMainWindowARB);

			//	if (!wglChoosePixelFormatARB(hDeviceARB, piAttribIList, pfAttribFList, 1, &pixelFormat, &numFormats)) {
			//		DestroyWindow(hMainWindowARB);
			//		throw std::exception("!wglChoosePixelFormatARB(...)");
			//	}
			//	if (!DescribePixelFormat(hDeviceARB, pixelFormat, sizeof(pfd), &pfd)) {
			//		DestroyWindow(hMainWindowARB);
			//		throw std::exception("!DescribePixelFormat(...)");
			//	}
			//	if (!SetPixelFormat(hDeviceARB, pixelFormat, &pfd)) {
			//		DestroyWindow(hMainWindowARB);
			//		throw std::exception("!SetPixelFormat(...)");
			//	}

			//	wglMakeCurrent(nullptr, nullptr);
			//	wglDeleteContext(hGLcontext);
			//	DestroyWindow(hMainWindow);
			//	hMainWindow = hMainWindowARB;
			//	hDevice = hDeviceARB;
			//	ShowWindow(hMainWindow, SW_SHOW);
			//	UpdateWindow(hMainWindow);

			//	this->hGLcontext = wglCreateContext(this->hDevice);
			//	wglMakeCurrent(this->hDevice, this->hGLcontext);
			//}
		}

		virtual ~window_opengl() noexcept override {
			if (this->context) {
				ReleaseDC(this->handler, this->context);
			}
		}
	};

	int waitkey(std::chrono::milliseconds dur = std::chrono::minutes(10)) {
		MSG msg = { 0 };
		auto time0 = std::chrono::steady_clock::now();
		while (true) {
			if ((std::chrono::steady_clock::now() - time0) > dur) {
				break;
			}
			if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
				if ( msg.message == WM_KEYDOWN ) {
					return (int)msg.wParam;
				}
			}
		}

		return 0;
	}

#ifdef _OPENGL_LIBRARY_HIGHLEVEL_
	inline HGLRC createGLRC(HDC context) {
		return wglCreateContext(context);
	}

	inline HGLRC createGLRC(window_opengl& window) {
		return wglCreateContext(window.context);
	}

	inline BOOL deleteGLRC(HGLRC opengl_context) {
		return wglDeleteContext(opengl_context);
	}

	inline BOOL bindGLRC(HDC context, HGLRC__& opengl_context) {
		return wglMakeCurrent(context, &opengl_context);
	}

	inline BOOL bindGLRC(window_opengl& window, HGLRC__& opengl_context) {
		return wglMakeCurrent(window.context, &opengl_context);
	}

	inline BOOL unbindGLRC() {
		return wglMakeCurrent(nullptr, nullptr);
	}

	template<bool __swapbuffers = true>
	BOOL imshow(window_opengl& target, HGLRC__& ctx, const GLimage& img, const std::string& imgexpr = "color = texture(image,texcoord);") {
		assert( target.context );
		HDC origin_context = wglGetCurrentDC();
		HGLRC origin_opengl_context = wglGetCurrentContext();
		wglMakeCurrent(target.context, &ctx);
		auto gl = GLlibrary(L"opengl32.dll");
		GLboolean is_blended = gl.IsEnabled(GL_BLEND);
		GLboolean is_depthtested = gl.IsEnabled(GL_DEPTH_TEST);
		GLint origin_viewport[4]; gl.GetIntegerv(GL_VIEWPORT, origin_viewport);
		GLboolean is_scissortested = gl.IsEnabled(GL_SCISSOR_TEST);
		GLint origin_scissorbox[4]; if (is_scissortested) { gl.GetIntegerv(GL_SCISSOR_BOX, origin_scissorbox); }
		gl.Disable(GL_BLEND);
		gl.Disable(GL_DEPTH_TEST);
		gl.Viewport(0, 0, target.window().get_clientsize().first, target.window().get_clientsize().second);
		gl.Disable(GL_SCISSOR_TEST);

		constexpr const char* vertex_shader_source = 
			"#version 450 core\n"

			"const vec4 VERTICES[3] = vec4[3](\n"
				"vec4(-1,-1,-1,1),\n"
				"vec4(+3,-1,-1,1),\n"
				"vec4(-1,+3,-1,1)\n"
			");\n"

			"out vec2 texcoord;\n"
			"void main(){\n"
				"gl_Position = VERTICES[gl_VertexID];\n"
				"texcoord = VERTICES[gl_VertexID].xy*0.5+0.5;\n"
			"}";
		std::string fragment_shader_source = 
			"#version 450 core\n"
			"layout(binding = 0) uniform sampler2D image;\n"
		
			"in vec2 texcoord;\n"
			"out vec4 color;\n"
			"void main() {\n"
				+ imgexpr +
			"}";
		GLuint vertex_shader = gl.CreateShader(GL_VERTEX_SHADER);
		GLuint fragment_shader = gl.CreateShader(GL_FRAGMENT_SHADER);
		const char* vt[1] = { vertex_shader_source };
		gl.ShaderSource(vertex_shader, 1, vt, nullptr);
		gl.CompileShader(vertex_shader);
		const char* ft[1] = { fragment_shader_source.c_str()};
		gl.ShaderSource(fragment_shader, 1, ft, nullptr);
		gl.CompileShader(fragment_shader);
		GLuint program = gl.CreateProgram();
		gl.AttachShader(program, vertex_shader);
		gl.AttachShader(program, fragment_shader);
		gl.LinkProgram(program);
		gl.DeleteShader(vertex_shader);
		gl.DeleteShader(fragment_shader);
		GLuint sampler;
		gl.GenSamplers(1, &sampler);
		gl.SamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		gl.SamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		gl.SamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
		gl.UseProgram(program);
		gl.ActiveTexture(GL_TEXTURE0);
		gl.BindTexture(GL_TEXTURE_2D, (GLuint)reinterpret_cast<size_t>(&img));
		gl.BindSampler(0, sampler);
		gl.DrawArrays(GL_TRIANGLES, 0, 3);

		gl.DeleteSamplers(1, &sampler);
		gl.DeleteProgram(program);
		gl.Finish();

		if (is_blended) { gl.Enable(GL_BLEND); }
		if (is_depthtested) { gl.Enable(GL_DEPTH_TEST); }
		gl.Viewport(origin_viewport[0], origin_viewport[1], origin_viewport[2], origin_viewport[3]);
		if (is_scissortested) { gl.Enable(GL_SCISSOR_TEST); gl.Scissor(origin_scissorbox[0], origin_scissorbox[1], origin_scissorbox[2], origin_scissorbox[3]); }
		wglMakeCurrent(origin_context, origin_opengl_context);
		
		if constexpr (__swapbuffers) {
			return SwapBuffers(target.context);
		} else {
			return true;
		}
		/*((window*)target.handler)->flash();
		((window*)target.handler)->update();*/
	}
#endif
}// end of namespace wex