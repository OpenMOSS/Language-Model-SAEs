//  @ts-check

import { tanstackConfig } from '@tanstack/eslint-config'

export default [
  ...tanstackConfig,
  {
    rules: {
      'no-shadow': 'off',
      '@typescript-eslint/no-unnecessary-condition': 'off',
      '@typescript-eslint/array-type': ['error', { default: 'array' }],
    },
  },
]
