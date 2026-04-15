#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <sstream>
#include <cstring>
#include <mutex>
#include <set>
#include <memory>

#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#define PLUGIN_CHECK_CUDA(call)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CUASSERT(status_)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            char const* msg = cudaGetErrorString(s_);                                                                  \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    } while (0)

#define GET_MACRO(_1, _2, NAME, ...) NAME

#define PLUGIN_VALIDATE(...) GET_MACRO(__VA_ARGS__, PLUGIN_VALIDATE_MSG, PLUGIN_VALIDATE_DEFAULT, )(__VA_ARGS__)

#define PLUGIN_VALIDATE_DEFAULT(condition)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, #condition);                            \
        }                                                                                                              \
    } while (0)

#define PLUGIN_VALIDATE_MSG(condition, msg)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                   \
        }                                                                                                              \
    } while (0)

#define PLUGIN_ASSERT(assertion)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    } while (0)

#define PLUGIN_FAIL(msg)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::plugin::reportAssertion(msg, __FILE__, __LINE__);                                                    \
    } while (0)

#define PLUGIN_ERROR(msg)                                                                                              \
    {                                                                                                                  \
        nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                       \
    }

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{

namespace pluginInternal
{

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

} // namespace pluginInternal

namespace plugin
{

template <ILogger::Severity kSeverity>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int32_t sync() override;
    };

    Buf buffer;
    std::mutex mLogStreamMutex;

public:
    std::mutex& getMutex()
    {
        return mLogStreamMutex;
    }
    LogStream()
        : std::ostream(&buffer){};
};

class TRTException : public std::exception
{
public:
    TRTException(char const* fl, char const* fn, int32_t ln, int32_t st, char const* msg, char const* nm)
        : file(fl)
        , function(fn)
        , line(ln)
        , status(st)
        , message(msg)
        , name(nm)
    {
    }
    virtual void log(std::ostream& logStream) const;
    void setMessage(char const* msg)
    {
        message = msg;
    }

protected:
    char const* file{nullptr};
    char const* function{nullptr};
    int32_t line{0};
    int32_t status{0};
    char const* message{nullptr};
    char const* name{nullptr};
};

class CudaError : public TRTException
{
public:
    CudaError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

class PluginError : public TRTException
{
public:
    PluginError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Plugin")
    {
    }
};

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;

void caughtError(std::exception const& e);

void throwCudaError(char const* file, char const* function, int32_t line, int32_t status, char const* msg);

void throwPluginError(char const* file, char const* function, int32_t line, int32_t status, char const* msg);

void reportValidationFailure(char const* msg, char const* file, int32_t line);

void reportAssertion(char const* msg, char const* file, int32_t line);

void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc);

template <typename Type, typename BufferType>
void write(BufferType*& buffer, Type const& val)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

template <typename OutType, typename BufferType>
OutType read(BufferType const*& buffer)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

size_t dataTypeSize(nvinfer1::DataType dtype);

} // namespace plugin
} // namespace nvinfer1

#endif // COMMON_H
