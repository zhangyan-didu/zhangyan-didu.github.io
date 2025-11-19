import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeelingList = [
  {
    title: '输出',
    description: '没有输出的输入，有意义吗？',
    date: '2025-11-08',
    link: '/docs/Feeling/2025-11-08',
  }
];

export default function FeelingsPage(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Feelings"
      description="抓住一缕烟">
      <div className={styles.container}>
        <div className={styles.header}>
          <Heading as="h1">Feelings</Heading>
          <p className={styles.subtitle}>抓住一缕烟</p>
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