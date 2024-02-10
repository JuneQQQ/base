import axios from 'axios';
import type {RouteRecordNormalized} from 'vue-router';
import {UserState} from '@/store/modules/user/types';


export interface LoginData {
    username?: string;
    password?: string;
    phone?: string;
    code?: number;
}


export function login(data: LoginData) {
    return axios.post<string>('/api/user/login', data);
}

export function sendCode(phone: number) {
    return axios.get<void>('/api/user/send-code', {params: {phone}});
}

export function logout() {
    return axios.post<void>('/api/user/logout');
}

export function getUserInfo() {
    return axios.post<UserState>('/api/user/info');
}

export function getMenuList() {
    return axios.post<RouteRecordNormalized[]>('/api/user/menu');
}
