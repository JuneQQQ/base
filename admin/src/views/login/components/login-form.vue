<template>
  <div class="login-form-wrapper">
    <div class="login-form-title">欢迎使用12307售票系统</div>
    <div class="login-form-sub-title">请登录</div>
    <br><br>
    <!--    <div class="login-form-error-msg">{{ errorMessage }}</div>-->
    <a-radio-group v-model="userInfo.loginBy" type="button">
      <a-radio value="password">密码登录</a-radio>
      <a-radio value="code">验证码登录</a-radio>
    </a-radio-group>
    <br>
    <br>
    <a-form
        ref="loginForm"
        :model="userInfo"
        auto-label-width
        layout="vertical"
        class="login-form"
        @submit="handleSubmit"
    >
      <a-form-item
          field="phone"
          :rules="[{ required: true, message: '手机号格式错误'}]"
          :validate-trigger="['change', 'blur']"
          hide-label
      >
        <a-input
            v-model="userInfo.phone"
            placeholder='请输入手机号'
        >
          <template #prefix>
            <icon-user/>
          </template>
        </a-input>
      </a-form-item>
      <a-form-item
          field="password"
          :rules="[{ required: true, message: '密码错误'}]"
          :validate-trigger="['change', 'blur']"
          hide-label
          v-if="userInfo.loginBy==='password'"
      >
        <a-input-password
            v-model="userInfo.password"
            placeholder="请输入密码"
            allow-clear
        >
          <template #prefix>
            <icon-lock/>
          </template>
        </a-input-password>

      </a-form-item>

      <a-form-item
          field="code"
          hide-asterisk
          class="verification-code-form-item"
          :rules="[
        {required:true,message:'验证码不正确'},
        {minLength:4,maxLength:4, message:'验证码不正确'},
        { match: /^\d+$/, message: '必须是数字' },
      ]"
          v-if="userInfo.loginBy==='code'"
      >
        <a-verification-code v-model="userInfo.code" style="width: 100px" :length=4></a-verification-code>
        <a-button size="medium" @click="sendCode(userInfo.phone)">{{ codeButton }}</a-button>
      </a-form-item>

      <a-space :size="16" direction="vertical">
        <div class="login-form-password-actions" v-show="userInfo.loginBy==='password'">
          <a-checkbox
              checked="rememberPassword"
              :model-value="loginConfig.rememberPassword"
              @change="setRememberPassword as any"
          >
            记住密码
          </a-checkbox>
          <a-link>忘记密码</a-link>
        </div>
        <a-button type="primary" html-type="submit" long :loading="loading">
          登录
        </a-button>
        <a-button type="text" long class="login-form-register-btn">
          注册账号
        </a-button>
      </a-space>
    </a-form>
  </div>
</template>

<script lang="ts" setup>
import {ref, reactive} from 'vue';
import {useRouter} from 'vue-router';
import {Message} from '@arco-design/web-vue';
import {ValidatedError} from '@arco-design/web-vue/es/form/interface';
import {useI18n} from 'vue-i18n';
import {useStorage} from '@vueuse/core';
import {useUserStore} from '@/store';
import useLoading from '@/hooks/loading';
import {LoginData} from '@/api/user';

const router = useRouter();
const errorMessage = ref('');
const {loading, setLoading} = useLoading();
const userStore = useUserStore();
const {t} = useI18n()

const loginConfig = useStorage('login-config', {
  loginBy: 'password',
  phone: 0,

  // 登录方式一
  password: '',
  rememberPassword: false,
  // 登录方式二
  code: '',
});
const userInfo = reactive({
  password: loginConfig.value.password,
  phone: loginConfig.value.phone,
  code: loginConfig.value.code,
  loginBy: loginConfig.value.loginBy,
});

const codeButton = ref('请输入验证码')

const handleSubmit = async ({
                              errors,
                              values,
                            }: {
  errors: Record<string, ValidatedError> | undefined;
  values: Record<string, any>;
}) => {
  if (loading.value) return;
  if (!errors) {
    setLoading(true);
    try {

      await userStore.login(values as LoginData);

      const {redirect, ...othersQuery} = router.currentRoute.value.query;
      console.log(redirect)
      router.push({
        name: (redirect as string) || 'Select',
        query: {
          ...othersQuery,
        },
      });
      Message.success('登陆成功');
      const {rememberPassword, loginBy} = loginConfig.value;

      if (rememberPassword && loginBy === 'phone') {
        const {phone, password} = values;
        // TODO 实际生产环境需要进行加密存储。
        loginConfig.value.phone = phone;
        loginConfig.value.password = password;
      }
    } catch (err) {
      errorMessage.value = (err as Error).message;
    } finally {
      setLoading(false);
    }
  }
};
const setRememberPassword = (value: boolean) => {
  loginConfig.value.rememberPassword = value;
};
const sendCode = (phone: number) => {
  const regex = /^(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\d{8}$/;
  if (!regex.test(String(phone))) {
    Message.error("手机号格式不正确")
    return
  }

  if (codeButton.value === "请输入验证码") {
    // 初始状态||验证码到时
    codeButton.value = "60";
    let count = 60;
    const intervalId = setInterval(() => {
      if (count <= 0) {
        codeButton.value = "请输入验证码";
        clearInterval(intervalId);
        return;
      }
      codeButton.value = count.toString();
      count -= 1;
    }, 1000);
    // 发请求
    userStore.sendCode(phone);

    Message.success("验证码发送成功")
  } else {
    // 验证码还在计时，直接返回
  }
}

</script>

<style lang="less" scoped>
:deep(.arco-form-item .arco-form-item-content) {
  justify-content: space-between;
}

.login-form {
  &-wrapper {
    width: 320px;
  }

  &-title {
    color: var(--color-text-1);
    font-weight: 500;
    font-size: 24px;
    line-height: 32px;
  }

  &-sub-title {
    color: var(--color-text-3);
    font-size: 16px;
    line-height: 24px;
  }

  &-error-msg {
    height: 32px;
    color: rgb(var(--red-6));
    line-height: 32px;
  }

  &-password-actions {
    display: flex;
    justify-content: space-between;
  }

  &-register-btn {
    color: var(--color-text-3) !important;
  }
}
</style>
