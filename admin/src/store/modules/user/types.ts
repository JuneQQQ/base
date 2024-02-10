export type RoleType = '*' | '' | 'admin' | 'user';

export interface UserState {
    id?: string;
    nickname?: string;
    avatar?: string;
    email?: string;
    phone?: number;
    registrationDate?: string;
    createTime: string
    role: RoleType;
}
