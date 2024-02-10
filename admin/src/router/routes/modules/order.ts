import {DEFAULT_LAYOUT} from '../base';
import {AppRouteRecordRaw} from '../types';

const ORDER: AppRouteRecordRaw = {
    path: '/order',
    name: 'order',
    component: DEFAULT_LAYOUT,
    meta: {
        locale: 'menu.order',
        requiresAuth: true,
        icon: 'icon-desktop',
        order: 0,
    },
    children: [
        {
            path: 'select',
            name: 'Select',
            component: () => import('@/views/order/select/index.vue'),
            meta: {
                locale: 'menu.order.select',
                requiresAuth: true,
                roles: ['*'],
            },
        },
    ],
};

export default ORDER;
