#ifndef VO_NAVIGATION_VO_STATE_H
#define VO_NAVIGATION_VO_STATE_H

namespace vo_navigation {

enum VOState {
    STATE_NORMAL = 0,
    STATE_ODOM_UNHEALTHY = 1, 
    STATE_RECOVER_OUTSIDE_LANES = 2
};

} // namespace vo_navigation

#endif // VO_NAVIGATION_VO_STATE_H
