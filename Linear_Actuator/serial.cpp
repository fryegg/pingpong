#include "serial.h"

CSerial_ctr::CSerial_ctr(std::string port)
{
	hSerial = CreateFile(port.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hSerial == INVALID_HANDLE_VALUE)
	{
		std::cout << "error #1" << std::endl;
	}
	dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
	if (GetCommState(hSerial, &dcbSerialParams) == 0)
	{
		std::cout << "error #2" << std::endl;
	}

	dcbSerialParams.BaudRate = 100000;
	dcbSerialParams.ByteSize = 8;
	dcbSerialParams.StopBits = NOPARITY;
	dcbSerialParams.Parity = NOPARITY;
	if (SetCommState(hSerial, &dcbSerialParams) == 0)
	{
		std::cout << "error #3" << std::endl;
	}

	timeouts.ReadIntervalTimeout = 5;
	timeouts.ReadTotalTimeoutConstant = 5;
	timeouts.ReadTotalTimeoutMultiplier = 5;
	timeouts.WriteTotalTimeoutConstant = 5;
	timeouts.WriteTotalTimeoutMultiplier = 5;
	if (SetCommTimeouts(hSerial, &timeouts) == 0)
	{
		std::cout << "error #4" << std::endl;
	}
}


CSerial_ctr::~CSerial_ctr()
{
	if (CloseHandle(hSerial) == 0)
	{
		std::cout << "error #5" << std::endl;
	}
}

void CSerial_ctr::write_data(std::string buf)
{
	DWORD bytes_written, total_bytes_written = 0;
	char write_buffer[1024];
	strcpy(write_buffer, buf.c_str());
	int size = buf.length() + 1;
	if (!WriteFile(hSerial, (LPSTR)(LPCSTR)(write_buffer), size, &bytes_written, NULL))
	{
		std::cout << "Send data error" << std::endl;
	}
}

std::string CSerial_ctr::read_data()
{
	DWORD bytes_written = 0;
	char buf[1024];
	if (!ReadFile(hSerial, (LPSTR)(LPCSTR)(buf), 1024, &bytes_written, NULL))
	{
		std::cout << "read data error" << std::endl;
	}

	return std::string(buf);
}

std::wstring CSerial_ctr::s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}
