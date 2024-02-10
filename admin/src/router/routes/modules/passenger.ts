import {DEFAULT_LAYOUT} from '../base';
import {AppRouteRecordRaw} from '../types';

const PASSENGER: AppRouteRecordRaw = {
    path: '/passenger',
    name: 'passenger',
    component: DEFAULT_LAYOUT,
    meta: {
        locale: 'menu.passenger',
        requiresAuth: true,
        icon: 'icon-idcard',
        order: 0,
    },
    children: [
        {
            path: 'management',
            name: 'Management',
            component: () => import('@/views/passenger/management/index.vue'),
            meta: {
                locale: 'menu.passenger.management',
                requiresAuth: true,
                roles: ['*'],
            },
        },
    ],
};

export default PASSENGER;
