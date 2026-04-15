#include "common.h"

namespace nvinfer1
{

namespace plugin
{

ILogger* gLogger{};

template <ILogger::Severity tSeverity>
int32_t LogStream<tSeverity>::Buf::sync()
{
    std::string s = str();
    while (!s.empty() && s.back() == '\n')
    {
        s.pop_back();
    }
    if (gLogger != nullptr)
    {
        gLogger->log(tSeverity, s.c_str());
    }
    str("");
    return 0;
}

LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;

void caughtError(std::exception const& e)
{
    gLogError << e.what() << std::endl;
}

void throwCudaError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudaError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

void throwPluginError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    PluginError error(file, function, line, status, msg);
    reportValidationFailure(msg, file, line);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

void reportValidationFailure(char const* msg, char const* file, int32_t line)
{
    std::ostringstream stream;
    stream << "Validation failed: " << msg << "\n" << file << ':' << line << "\n";
#ifdef COMPILE_VFC_PLUGIN
    ILogger* logger = getPluginLogger();
    if (logger != nullptr)
    {
        logger->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    }
#else
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
#endif
}

void reportAssertion(char const* msg, char const* file, int32_t line)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << "\n"
           << file << ':' << line << "\n"
           << "Aborting..."
           << "\n";
#ifdef COMPILE_VFC_PLUGIN
    ILogger* logger = getPluginLogger();
    if (logger != nullptr)
    {
        logger->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    }
#else
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
#endif
    PLUGIN_CUASSERT(cudaDeviceReset());
    exit(EXIT_FAILURE);
}

void TRTException::log(std::ostream& logStream) const
{
    logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
    if (message != nullptr)
    {
        logStream << " (" << message << ")";
    }
    logStream << std::endl;
}

void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        requiredFieldNames.erase(fc->fields[i].name);
    }
    if (!requiredFieldNames.empty())
    {
        std::stringstream msg{};
        msg << "PluginFieldCollection missing required fields: {";
        char const* separator = "";
        for (auto const& field : requiredFieldNames)
        {
            msg << separator << field;
            separator = ", ";
        }
        msg << "}";
        std::string msg_str = msg.str();
        PLUGIN_ERROR(msg_str.c_str());
    }
}

size_t dataTypeSize(const DataType dtype)
{
    switch (dtype)
    {
    case DataType::kINT8: return sizeof(char);
    case DataType::kHALF: return sizeof(short);
    case DataType::kFLOAT: return sizeof(float);
    default: PLUGIN_FAIL("Unsupported data type");
    return 0;
    }
}

} // namespace plugin
} // namespace nvinfer1
