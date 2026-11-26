/** 表单与表格的公共配置工厂
 *
 * 这两个形态在 36 处表单校验、5 处表格分页里各写了一遍。
 * 收敛到这里后，调整每页条数或校验风格只需改一处。
 */

import type { Rule } from "antd/es/form";
import type { TablePaginationConfig } from "antd";

/** 必填校验规则，文案由调用方给出（"请输入名称""请选择运行时间"等）
 *
 * 返回单个规则而非数组，便于和多条件校验并列：
 *   rules={[requiredRule("请输入新密码"), { min: 6, message: "..." }]}
 */
export const requiredRule = (message: string): Rule => ({ required: true, message });

/** 列表页的统一分页配置，unit 为量词，如"个知识库""个用户" */
export const simplePagination = (unit: string): TablePaginationConfig => ({
  pageSize: 10,
  showTotal: (total: number) => `共 ${total} ${unit}`,
});
