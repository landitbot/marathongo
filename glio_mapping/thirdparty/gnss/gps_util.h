#ifndef COORD_CONVERTER_H
#define COORD_CONVERTER_H 
 
#include <cmath>
#include <algorithm>
#include <iostream>
 #include <gdal/ogr_spatialref.h> // GDAL头文件
constexpr double PI = 3.14159265358979323846;
constexpr double PIX = PI * 3000.0 / 180.0;
constexpr double EE = 0.00669342162296594323;
constexpr double A = 6378245.0;
 
namespace GPSUtil {
 
// 辅助函数：经纬度变换核心算法
inline void transformLat(double lng, double lat, double& ret) {
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * std::sqrt(std::fabs(lng));
    ret += (20.0 * std::sin(6.0 * lng * PI) + 20.0 * std::sin(2.0 * lng * PI)) * 2.0 / 3.0;
    ret += (20.0 * std::sin(lat * PI) + 40.0 * std::sin(lat / 3.0 * PI)) * 2.0 / 3.0;
    ret += (160.0 * std::sin(lat / 12.0 * PI) + 320.0 * std::sin(lat * PI / 30.0)) * 2.0 / 3.0;
}
 
inline void transformLng(double lng, double lat, double& ret) {
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * std::sqrt(std::fabs(lng));
    ret += (20.0 * std::sin(6.0 * lng * PI) + 20.0 * std::sin(2.0 * lng * PI)) * 2.0 / 3.0;
    ret += (20.0 * std::sin(lng * PI) + 40.0 * std::sin(lng / 3.0 * PI)) * 2.0 / 3.0;
    ret += (150.0 * std::sin(lng / 12.0 * PI) + 300.0 * std::sin(lng / 30.0 * PI)) * 2.0 / 3.0;
}
 
// 边界检测
inline bool outOfChina(double lat, double lon) {
    return (lon < 72.004 || lon > 137.8347) || 
           (lat < 0.8293 || lat > 55.8271);
}
 
// 坐标系基础转换
inline void transform(double lat, double lon, double& out_lat, double& out_lon) {
    if (outOfChina(lat, lon)) {
        out_lat = lat;
        out_lon = lon;
        return;
    }
    
    double dLat, dLon;
    transformLat(lon - 105.0, lat - 35.0, dLat);
    transformLng(lon - 105.0, lat - 35.0, dLon);
    
    double radLat = lat / 180.0 * PI;
    double magic = std::sin(radLat);
    magic = 1 - EE * magic * magic;
    double sqrtMagic = std::sqrt(magic);
    
    dLat = (dLat * 180.0) / ((A * (1 - EE)) / (magic * sqrtMagic) * PI);
    dLon = (dLon * 180.0) / (A / sqrtMagic * std::cos(radLat) * PI);
    
    out_lat = lat + dLat;
    out_lon = lon + dLon;
}
 
// BD09 -> GCJ02
inline void bd09_to_gcj02(double lng, double lat, double& out_lng, double& out_lat) {
    double x = lng - 0.0065;
    double y = lat - 0.006;
    double z = std::sqrt(x * x + y * y) - 0.00002 * std::sin(y * PIX);
    double theta = std::atan2(y, x) - 0.000003 * std::cos(x * PIX);
    out_lng = z * std::cos(theta);
    out_lat = z * std::sin(theta);
}
 
// GCJ02 -> BD09
inline void gcj02_to_bd09(double lng, double lat, double& out_lng, double& out_lat) {
    double z = std::sqrt(lng * lng + lat * lat) + 0.00002 * std::sin(lat * PIX);
    double theta = std::atan2(lat, lng) + 0.000003 * std::cos(lng * PIX);
    out_lng = z * std::cos(theta) + 0.0065;
    out_lat = z * std::sin(theta) + 0.006;
}
 
// GCJ02 -> WGS84
inline void gcj02_to_wgs84(double lng, double lat, double& out_lng, double& out_lat) {
    double dlat, dlng;
    transformLat(lng - 105.0, lat - 35.0, dlat);
    transformLng(lng - 105.0, lat - 35.0, dlng);
    
    double radlat = lat / 180.0 * PI;
    double magic = std::sin(radlat);
    magic = 1 - EE * magic * magic;
    double sqrtmagic = std::sqrt(magic);
    
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * PI);
    dlng = (dlng * 180.0) / (A / sqrtmagic * std::cos(radlat) * PI);
    
    out_lng = lng - dlng;
    out_lat = lat - dlat;
}
 
// WGS84 -> GCJ02 
inline void wgs84_to_gcj02(double lng, double lat, double& out_lng, double& out_lat) {
    double dlat, dlng;
    transformLat(lng - 105.0, lat - 35.0, dlat);
    transformLng(lng - 105.0, lat - 35.0, dlng);
    
    double radlat = lat / 180.0 * PI;
    double magic = std::sin(radlat);
    magic = 1 - EE * magic * magic;
    double sqrtmagic = std::sqrt(magic);
    
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * PI);
    dlng = (dlng * 180.0) / (A / sqrtmagic * std::cos(radlat) * PI);
    
    out_lng = lng + dlng;
    out_lat = lat + dlat;
}
 
// MapBar -> WGS84 
inline void mapbar_to_wgs84(double lng, double lat, double& out_lng, double& out_lat) {
    lng = lng * 100000.0;
    lat = lat * 100000.0;
    
    lng = std::fmod(lng, 36000000.0);
    lat = std::fmod(lat, 36000000.0);
    
    double lng1 = lng - std::cos(lat / 100000.0) * lng / 18000.0 - std::sin(lng / 100000.0) * lat / 9000.0;
    double lat1 = lat - std::sin(lat / 100000.0) * lng / 18000.0 - std::cos(lng / 100000.0) * lat / 9000.0;
    
    double sign_lng = (lng > 0) ? 1.0 : -1.0;
    double sign_lat = (lat > 0) ? 1.0 : -1.0;
    
    double lng2 = lng - std::cos(lat1 / 100000.0) * lng1 / 18000.0 - std::sin(lng1 / 100000.0) * lat1 / 9000.0 + sign_lng;
    double lat2 = lat - std::sin(lat1 / 100000.0) * lng1 / 18000.0 - std::cos(lng1 / 100000.0) * lat1 / 9000.0 + sign_lat;
    
    out_lng = lng2 / 100000.0;
    out_lat = lat2 / 100000.0;
}
 
// WGS84 -> MapBar (迭代实现)
inline void wgs84_to_mapbar(double lng_w, double lat_w, double& out_lng_m, double& out_lat_m, int max_iter = 20, double tol = 1e-7) {
    // 初始值
    double lng_m = lng_w;
    double lat_m = lat_w;
    
    for (int i = 0; i < max_iter; ++i) {
        double lng_calc, lat_calc;
        mapbar_to_wgs84(lng_m, lat_m, lng_calc, lat_calc);
        
        // 计算误差 
        double d_lng = lng_w - lng_calc;
        double d_lat = lat_w - lat_calc;
        
        // 更新估计值
        lng_m += d_lng;
        lat_m += d_lat;
        
        // 检查收敛
        if (std::abs(d_lng) < tol && std::abs(d_lat) < tol) {
            break;
        }
    }
    
    out_lng_m = lng_m;
    out_lat_m = lat_m;
}
 
// BD09 -> WGS84
inline void bd09_to_wgs84(double lng, double lat, double& out_lng, double& out_lat) {
    double gcj_lng, gcj_lat;
    bd09_to_gcj02(lng, lat, gcj_lng, gcj_lat);
    gcj02_to_wgs84(gcj_lng, gcj_lat, out_lng, out_lat);
}
 
// WGS84 -> BD09
inline void wgs84_to_bd09(double lng, double lat, double& out_lng, double& out_lat) {
    double gcj_lng, gcj_lat;
    wgs84_to_gcj02(lng, lat, gcj_lng, gcj_lat);
    gcj02_to_bd09(gcj_lng, gcj_lat, out_lng, out_lat);
}
 
// MapBar -> GCJ02
inline void mapbar_to_gcj02(double lng, double lat, double& out_lng, double& out_lat) {
    double wgs_lng, wgs_lat;
    mapbar_to_wgs84(lng, lat, wgs_lng, wgs_lat);
    wgs84_to_gcj02(wgs_lng, wgs_lat, out_lng, out_lat);
}
 
// GCJ02 -> MapBar
inline void gcj02_to_mapbar(double lng, double lat, double& out_lng, double& out_lat) {
    double wgs_lng, wgs_lat;
    gcj02_to_wgs84(lng, lat, wgs_lng, wgs_lat);
    wgs84_to_mapbar(wgs_lng, wgs_lat, out_lng, out_lat);
}
 
// MapBar -> BD09 
inline void mapbar_to_bd09(double lng, double lat, double& out_lng, double& out_lat) {
    double wgs_lng, wgs_lat;
    mapbar_to_wgs84(lng, lat, wgs_lng, wgs_lat);
    wgs84_to_bd09(wgs_lng, wgs_lat, out_lng, out_lat);
}
 
// BD09 -> MapBar
inline void bd09_to_mapbar(double lng, double lat, double& out_lng, double& out_lat) {
    double wgs_lng, wgs_lat;
    bd09_to_wgs84(lng, lat, wgs_lng, wgs_lat);
    wgs84_to_mapbar(wgs_lng, wgs_lat, out_lng, out_lat);
}
/**
 * @brief 计算所属的3度带带号 
 */
inline int lon_to_3deg_zone(double lon) {
    return static_cast<int>((lon + 1.5) / 3.0);
}
 
/**
 * @brief 将 WGS84 经纬度转换为 CGCS2000 3度带投影坐标 
 */
inline bool wgs84_to_cgcs2000_3deg(double lon, double lat, double& x, double& y, int lon0=117) {
    // 1. 计算带号和中央经线 
    // zone = lon_to_3deg_zone(lon) * 3;
    // double lon0 = zone;
 
    // 2. 定义源坐标系: WGS84 (EPSG:4326)
    OGRSpatialReference oSrcSRS;
    oSrcSRS.importFromEPSG(4326);
    // 【关键】设置坐标轴顺序为传统GIS顺序，对应 Python 的 always_xy=True 
    oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 
    // 3. 定义目标坐标系: CGCS2000 / 3-degree Gauss-Kruger zone 
    OGRSpatialReference oDstSRS;
 
    // ---------------------------------------------------------
    // 方法 A: 使用 GDAL API 构建 (推荐，更易读，参数明确)
    // ---------------------------------------------------------
    // oDstSRS.SetTM(0.0, lon0, 1.0, 500000.0, 0.0);
    // oDstSRS.SetGeogCS("CGCS2000", "China 2000", "CGCS2000", 
    //                   6378137.0, 298.257222101, 
    //                   "Greenwich", 0.0, 
    //                   "degree", 0.0174532925199433);
    // oDstSRS.SetLinearUnits("meter", 1.0);
 
    // ---------------------------------------------------------
    // 方法 B: 使用 Proj4 字符串 (完全复刻Python代码逻辑)
    // ---------------------------------------------------------
    const char* proj4Str = "+proj=tmerc +lat_0=0 +lon_0=117 +k=1.0 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs";
    oDstSRS.importFromProj4(proj4Str);
    
    // 注意：使用 Proj4 字符串导入时，通常默认轴顺序即为，
    // 但为了保险起见，也可以显式设置目标坐标系的轴顺序：
    oDstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 
    // 4. 创建坐标转换器 
    OGRCoordinateTransformation* poCT = OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);
    if (poCT == nullptr) {
        std::cerr << "Error: Failed to create coordinate transformation." << std::endl;
        return false;
    }
 
    // 5. 执行转换 
    x = lon;
    y = lat;
    
    if (!poCT->Transform(1, &x, &y)) {
        std::cerr << "Error: Transformation failed." << std::endl;
        delete poCT;
        return false;
    }
 
    // 6. 清理资源 
    delete poCT;
    return true;
}
/**
 * @brief 将 CGCS2000 3度带投影坐标转换为 WGS84 经纬度 
 * @param x 输入东坐标
 * @param y 输入北坐标 
 * @param zone 输入中央经线 (单位：度，如 117)
 * @param lon 输出经度
 * @param lat 输出纬度
 * @return 是否成功
 */
inline bool cgcs2000_3deg_to_wgs84(double x, double y, double& lon, double& lat, int lon0=117) {
    // 1. 确定中央经线
    // double lon0 = static_cast<double>(zone);
 
    // 2. 定义源坐标系: CGCS2000 / 3-degree Gauss-Kruger zone
    OGRSpatialReference oSrcSRS;
    const char* proj4Str = "+proj=tmerc +lat_0=0 +lon_0=117 +k=1.0 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs";
    oSrcSRS.importFromProj4(proj4Str);
    oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 
    // 3. 定义目标坐标系: WGS84 (EPSG:4326)
    OGRSpatialReference oDstSRS;
    oDstSRS.importFromEPSG(4326);
    // 【关键】设置坐标轴顺序为传统GIS顺序，确保输出为
    oDstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 
    // 4. 创建坐标转换器 (反向：Src -> Dst)
    OGRCoordinateTransformation* poCT = OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);
    if (poCT == nullptr) {
        std::cerr << "Error: Failed to create coordinate transformation." << std::endl;
        return false;
    }
 
    // 5. 执行转换
    lon = x;
    lat = y;
 
    if (!poCT->Transform(1, &lon, &lat)) {
        std::cerr << "Error: Transformation failed." << std::endl;
        delete poCT;
        return false;
    }
 
    // 6. 清理资源 
    delete poCT;
    return true;
}
inline bool ConvertLonLatToXY(double lon, double lat, double& x, double& y, int lon0=117){
    double out_lng, out_lat;
    wgs84_to_gcj02(lon, lat, out_lng, out_lat);
    wgs84_to_cgcs2000_3deg(out_lng, out_lat, x, y, lon0);
    // std::cout  << std::fixed << std::setprecision(6) << "x:" << x << ",y:" << y << std::endl;
    return true;
}
  
} // namespace GPSUtil
 
#endif // COORD_CONVERTER_H