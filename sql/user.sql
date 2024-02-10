create table train.user
(
    id          bigint       not null comment 'id'
        primary key,
    nickname    varchar(32)  null comment '昵称',
    avatar      varchar(255) null comment '头像链接',
    phone       varchar(11)  null comment '手机号',
    password    varchar(32)  null,
    create_time datetime     null,
    update_time datetime     null,
    constraint phone_unique
        unique (phone)
)
comment '会员';

