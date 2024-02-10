import axios from 'axios';
import type {AxiosRequestConfig, AxiosResponse} from 'axios';
import {Message, Modal} from '@arco-design/web-vue';
import {useUserStore} from '@/store';
import {getToken} from '@/utils/auth';

export interface HttpResponse<T = unknown> {
    code: number;
    msg: string;
    data: T;
}

if (import.meta.env.VITE_API_BASE_URL) {
    axios.defaults.baseURL = import.meta.env.VITE_API_BASE_URL;
}

axios.interceptors.request.use(
    (config: AxiosRequestConfig) => {
        // let each request carry token
        // this example using the JWT token
        // Authorization is a custom headers key
        // please modify it according to the actual situation
        const token = getToken();
        if (token) {
            if (!config.headers) {
                config.headers = {};
            }
            config.headers.token = token;
        }
        return config;
    },
    (error) => {
        // do something
        return Promise.reject(error);
    }
);
// add response interceptors
axios.interceptors.response.use(
    (response: AxiosResponse<HttpResponse>) => {
        const res = response.data;
        if (res.code !== 200) {
            Message.error({
                content: res.msg || 'Error',
                duration: 5 * 1000,
            });
            // TODO
            // 50008: Illegal token; 50012: Other clients logged in; 50014: Token expired;
            if (
                [1004, 1005, 1006].includes(res.code) &&
                response.config.url !== '/api/user/info'
            ) {
                Modal.error({
                    title: '提示',
                    content:
                        '登录信息已过期',
                    okText: '去登录',
                    async onOk() {
                        const userStore = useUserStore();

                        await userStore.logout();
                        window.location.reload();
                    },
                });
            }
            return Promise.reject(new Error(res.msg || 'Error'));
        }
        return res;
    },
    (error) => {
        Message.error({
            content: error.msg || 'Request Error',
            duration: 5 * 1000,
        });
        return Promise.reject(error);
    }
);
