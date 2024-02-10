import {DEFAULT_LAYOUT} from '../base';
import {AppRouteRecordRaw} from '../types';

const TICKET: AppRouteRecordRaw = {
    path: '/ticket',
    name: 'ticket',
    component: DEFAULT_LAYOUT,
    meta: {
        locale: 'menu.ticket',
        requiresAuth: true,
        icon: 'icon-calendar',
        order: 0,
    },
    children: [
        {
            path: 'ticket',
            name: 'Ticket',
            component: () => import('@/views/ticket/buy/index.vue'),
            meta: {
                locale: 'menu.ticket.buy',
                requiresAuth: true,
                roles: ['*'],
            },
        },
    ],
};

export default TICKET;
