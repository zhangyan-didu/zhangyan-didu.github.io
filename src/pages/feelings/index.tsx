import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeelingList = [
  {
    title: '秋日的感悟',
    description: '秋风起时，万物凋零，但也带来了新的思考',
    date: '2024-09-15',
    link: '#',
  },
  {
    title: '编程与人生',
    description: '代码的逻辑与人生的哲理，有着相似的韵律',
    date: '2024-09-10',
    link: '#',
  },
  {
    title: '宁静的夜晚',
    description: '夜深人静时，思绪如流水般涌动',
    date: '2024-09-05',
    link: '#',
  },
  {
    title: '学习的快乐',
    description: '掌握新知识时的那种成就感，是无价的',
    date: '2024-09-01',
    link: '#',
  },
];

export default function FeelingsPage(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Feelings"
      description="记录生活中的美好瞬间和心灵独白">
      <div className={styles.container}>
        <div className={styles.header}>
          <Heading as="h1">Feelings</Heading>
          <p className={styles.subtitle}>心情感悟 · 生活中的美好瞬间和心灵独白</p>
        </div>

        <div className={styles.grid}>
          {FeelingList.map((feeling, index) => (
            <div key={index} className={styles.card}>
              <div className={styles.cardContent}>
                <Heading as="h3">{feeling.title}</Heading>
                <p className={styles.description}>{feeling.description}</p>
                <div className={styles.date}>{feeling.date}</div>
                <Link className="button button--secondary" to={feeling.link}>
                  阅读更多
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
}